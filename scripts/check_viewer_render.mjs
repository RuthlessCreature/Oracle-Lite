// Isolated application test with software WebGL and a temporary browser profile.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire((process.env.VIEWER_PLAYWRIGHT_DIR||'/tmp/oracle-viewer-ui-test')+'/package.json');
const {chromium}=require('playwright');
const origin=process.env.VIEWER_TEST_ORIGIN||'http://127.0.0.1:7870';
const project=process.env.VIEWER_TEST_PROJECT||'9721ec1c27e045aa955028e8a2807016';
const output=process.env.VIEWER_RENDER_REPORT||'../reports/oracle-viewer/render';await fs.mkdir(output,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium-browser',headless:true,
 args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1680,height:1050},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.goto(origin+`/viewer?project=${encodeURIComponent(project)}&render-test=1`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>window.__oracleViewerDiagnostics?.().mode==='3d' && window.__oracleViewerDiagnostics?.().triangles>0,{timeout:90000});
 const snapshot=()=>page.evaluate(()=>window.__oracleViewerDiagnostics());
 const first=await snapshot();assert.equal(first.mode,'3d');assert.equal(first.meshCount,107);
 assert.equal(first.faceCount,5221);assert.ok(first.triangles>271171&&first.triangles<=1500000);assert.equal(first.drawCalls,107);
 assert.ok(first.cameraFar/first.cameraNear<10000,'camera depth range should be tight enough for thin assembly details');
 assert.equal(await page.locator('#marker-mode').inputValue(),'active');
 assert.equal(first.environment,true);assert.equal(await page.locator('#geometry-empty').isVisible(),false);
 await page.locator('#marker-mode').selectOption('active');
 await page.screenshot({path:output+'/viewer-isometric.png'});
 const canvas=page.locator('#canvas-host canvas'),bounds=await canvas.boundingBox();
 const cx=bounds.x+bounds.width*.5,cy=bounds.y+bounds.height*.5;
 await page.mouse.move(cx,cy);await page.mouse.down();await page.mouse.move(cx+100,cy+65,{steps:12});await page.mouse.up();
 await page.waitForTimeout(600);const rotated=await snapshot();assert.notDeepEqual(rotated.camera,first.camera);
 const distance=s=>Math.hypot(...s.camera.map((v,i)=>v-s.target[i]));
 await page.mouse.wheel(0,-320);await page.waitForTimeout(600);assert.ok(distance(await snapshot())<distance(rotated));
 await page.locator('#top').click();await page.waitForTimeout(600);await page.screenshot({path:output+'/viewer-top.png'});
 const b=await canvas.boundingBox();let picked=false;
 outer:for(const y of [.14,.16,.18,.82,.86,.88,.3,.6])for(const x of [.5,.4,.6,.35,.36,.37,.38,.62,.63,.64,.65]){
  await page.mouse.click(b.x+b.width*x,b.y+b.height*y);
  if((await snapshot()).selected.length){picked=true;break outer;}
 }
 assert.equal(picked,true);assert.match((await snapshot()).selected[0],/:instance\d+:face\d+$/);
 assert.equal(await page.locator('#bind').isDisabled(),false);
 await page.locator('#transparent').click();assert.equal((await snapshot()).transparent,true);
 await page.locator('#transparent').click();await page.locator('#clip').check();
 await page.locator('#clip-value').fill('65');await page.locator('#clip-value').dispatchEvent('input');
 assert.equal((await snapshot()).clipped,true);
 await page.screenshot({path:output+'/viewer-section.png'});
 await page.locator('#clip').uncheck();await page.locator('#fit').click();
 await page.locator('#view-source').click();assert.equal((await snapshot()).mode,'source');
 await page.locator('#source-image').evaluate(img=>img.decode());assert.equal(await page.locator('#source-image').isVisible(),true);
 await page.locator('#view-3d').click();assert.equal((await snapshot()).mode,'3d');
 assert.deepEqual(errors,[]);
 await fs.writeFile(output+'/render-test.json',JSON.stringify({status:'passed',browser:await browser.version(),
  rendering:'Chromium software WebGL / SwiftShader',first,rotated,checks:['refined actual X_T mesh','assembly instances',
   'body draw-call merging','studio environment','orbit','zoom','top view','source face picking',
   'preview visual annotation enabled','transparency','section','source/3D navigation'],errors},null,2));
 console.log(`PASS: refined X_T WebGL render, 107 instances / 5,221 faces / ${first.triangles.toLocaleString()} triangles; orbit, zoom, picking, transparency, section and source navigation.`);
}finally{await browser.close();}
