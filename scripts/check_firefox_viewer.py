"""W3C WebDriver regression against installed Firefox, using an isolated profile.
Start geckodriver on 4445 with a snap-visible temporary profile root first.
"""
import json,time,base64,os
from pathlib import Path
import requests
s=requests.Session();s.trust_env=False
root='http://127.0.0.1:4445';sid=None
headed=os.environ.get('ORACLE_FIREFOX_HEADED')=='1'
report_prefix='firefox-headed' if headed else 'firefox'
out=Path('../reports/oracle-viewer/firefox');out.mkdir(parents=True,exist_ok=True)
def call(method,path,data=None):
 r=s.request(method,root+path,json=data,timeout=90);v=r.json().get('value')
 if r.status_code>=400:raise RuntimeError(str(v))
 return v
try:
 session=call('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':[] if headed else ['-headless'],'prefs':{'webgl.disabled':False,'webgl.force-enabled':True}}}}})
 sid=session['sessionId'];prefix='/session/'+sid
 def js(script,*args):return call('POST',prefix+'/execute/sync',{'script':script,'args':list(args)})
 def wait(script,timeout=90):
  start=time.monotonic()
  while time.monotonic()-start<timeout:
   value=js(script)
   if value:return value
   time.sleep(.15)
  raise TimeoutError(script)
 call('POST',prefix+'/window/rect',{'width':1440,'height':1000})
 call('POST',prefix+'/url',{'url':'http://127.0.0.1:7870/viewer?project=9721ec1c27e045aa955028e8a2807016&render-test=1'})
 first=wait("return window.__oracleViewerDiagnostics?.().mode==='3d' && window.__oracleViewerDiagnostics().meshCount===107 && window.__oracleViewerDiagnostics().triangles>1000000 && window.__oracleViewerDiagnostics();")
 print('Firefox',session['capabilities']['browserVersion'],'CAD rendered',first['triangles'],flush=True)
 assert first['retainedSourceVertexArrays']==0
 time.sleep(2);idle1=js('return window.__oracleViewerDiagnostics();');time.sleep(2);idle2=js('return window.__oracleViewerDiagnostics();')
 assert idle2['renderedFrames']-idle1['renderedFrames']<=1,'idle scene must not redraw continuously'
 js("document.getElementById('open-workflow').click();")
 wait("return document.querySelector('[data-head]') || document.querySelector('#wf-steps button[data-step=\"4\"]');")
 js("document.querySelector('#wf-steps button[data-step=\"4\"]').click();")
 wait("return document.querySelector('[data-head]');")
 for i in range(12):
  js("const choices=[...document.querySelectorAll('[data-head]')];choices[arguments[0]%choices.length].click();",i)
  layout=wait('return window.__oracleViewerDiagnostics?.().planningObjectCount>=8 && window.__oracleViewerDiagnostics();')
  assert not layout['contextLost'];assert len([m for m in layout['markers'] if m['visible']])>=1
  if i==0:
   js("document.getElementById('simulate-light').click();document.getElementById('camera-preview').click();")
   wait('return window.__oracleViewerDiagnostics().previewOn && window.__oracleViewerDiagnostics().simulationOn;')
   png=call('GET',prefix+'/screenshot');(out/(report_prefix+'-camera-preview.png')).write_bytes(base64.b64decode(png))
   js("document.getElementById('camera-preview').click();document.getElementById('simulate-light').click();")
   png=call('GET',prefix+'/screenshot');(out/(report_prefix+'-layout.png')).write_bytes(base64.b64decode(png))
  js("document.getElementById('clear-planning').click();")
  clean=wait('return !window.__oracleViewerDiagnostics().planningHead && window.__oracleViewerDiagnostics().gpuResources.geometries<=110 && window.__oracleViewerDiagnostics();')
  assert clean['gpuResources']['textures']<=2
  assert clean['gpuResources']['programs']<=8
  js("document.getElementById('open-workflow').click();")
 print('12 layout/clear cycles passed',flush=True)
 # Exercise real resize high-DPI bounds and a context-loss recovery without crashing the tab.
 call('POST',prefix+'/window/rect',{'width':2560,'height':1600});js("document.getElementById('wf-close').click();")
 time.sleep(.5);resized=js('return window.__oracleViewerDiagnostics();')
 assert resized['drawingBuffer'][0]*resized['drawingBuffer'][1]<=1500000
 assert resized['retainedSourceVertexArrays']==0
 context_recovery=js("const gl=document.querySelector('#canvas-host canvas').getContext('webgl2');window.__testContextLoss=gl.getExtension('WEBGL_lose_context');if(window.__testContextLoss){window.__testContextLoss.loseContext();return true;}return false;")
 if context_recovery:
  wait('return window.__oracleViewerDiagnostics().contextLost;')
  js('window.__testContextLoss.restoreContext();delete window.__testContextLoss;')
  recovered=wait('return !window.__oracleViewerDiagnostics().contextLost && window.__oracleViewerDiagnostics().gpuResources.geometries===107 && window.__oracleViewerDiagnostics();')
  assert recovered['gpuResources']['textures']==1
 else:recovered=None
 gl_renderer=js("const gl=document.querySelector('#canvas-host canvas').getContext('webgl2'); const ext=gl.getExtension('WEBGL_debug_renderer_info');return gl.getParameter(ext?ext.UNMASKED_RENDERER_WEBGL:gl.RENDERER);")
 (out/(report_prefix+'-test.json')).write_text(json.dumps({'status':'passed','browser':session['capabilities']['browserVersion'],'mode':'isolated headed Firefox WebDriver' if headed else 'isolated headless Firefox WebDriver','gl_renderer':gl_renderer,'initial':first,'idle_frames_delta':idle2['renderedFrames']-idle1['renderedFrames'],'cycles':12,'context_recovery':context_recovery,'recovered':recovered,'layout':layout,'clean':clean,'resized':resized},indent=2))
 print('PASS Firefox: CAD, layout, repeated cleanup, idle rendering and drawing-buffer budget',flush=True)
finally:
 if sid:call('DELETE','/session/'+sid)
