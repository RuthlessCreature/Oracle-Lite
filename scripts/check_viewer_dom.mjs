// Application DOM regression checks; no real browser or visual/WebGL acceptance.
// Node 22+, jsdom 26.1.0 installed in VIEWER_JSDOM_DIR (default /tmp/oracle-viewer-ui-test).
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire(pathToFileURL((process.env.VIEWER_JSDOM_DIR||'/tmp/oracle-viewer-ui-test')+'/package.json'));
const {JSDOM,VirtualConsole}=require('jsdom');
const origin=process.env.VIEWER_TEST_ORIGIN||'http://127.0.0.1:7870';
const root=process.cwd(), html=await fs.readFile(root+'/oracle_viewer/static/index.html','utf8');
const dom=new JSDOM(html,{url:origin+'/viewer',virtualConsole:new VirtualConsole()});
const nativeFetch=globalThis.fetch;
Object.assign(globalThis,{window:dom.window,document:dom.window.document,location:dom.window.location,
  devicePixelRatio:1,ResizeObserver:class{observe(){}},requestAnimationFrame:()=>{},
  fetch:(url,options)=>nativeFetch(new URL(url,origin),options)});
// Exercise the graceful unavailable-WebGL branch while retaining real Three geometry classes.
const {mkdtemp}=await import('node:fs/promises');const temp=await mkdtemp('/tmp/oracle-viewer-dom-');
await fs.copyFile(root+'/oracle_viewer/static/vendor/three.module.js',temp+'/three.mjs');
await fs.copyFile(root+'/oracle_viewer/static/vendor/OrbitControls.js',temp+'/orbit.mjs');
let orbit=await fs.readFile(temp+'/orbit.mjs','utf8');await fs.writeFile(temp+'/orbit.mjs',orbit.replace("from 'three'","from './three.mjs'"));
await fs.copyFile(root+'/oracle_viewer/static/vendor/RoomEnvironment.js',temp+'/room.mjs');
let room=await fs.readFile(temp+'/room.mjs','utf8');await fs.writeFile(temp+'/room.mjs',room.replace("from 'three'","from './three.mjs'"));
let source=await fs.readFile(root+'/oracle_viewer/static/viewer.js','utf8');source=source.replace("from 'three'","from './three.mjs'").replace("from './vendor/OrbitControls.js'","from './orbit.mjs'").replace("from './vendor/RoomEnvironment.js'","from './room.mjs'");await fs.writeFile(temp+'/viewer.mjs',source);
await import(pathToFileURL(temp+'/viewer.mjs'));
async function until(test){for(let i=0;i<100;i++){if(test())return;await new Promise(r=>setTimeout(r,30));}throw Error('DOM update timed out');}
const $=id=>document.getElementById(id);
await until(()=>$('fai-title').textContent==='FAI 23');
assert.equal($('items').children.length,82);
assert.match($('methods').textContent,/CIR_N1_VISION/);
assert.match($('dependencies').textContent,/ALI_ABD/);
assert.match($('source-image').src,/pages\/33.png$/);
assert.equal($('bind').disabled,true);
assert.match($('talker').href,/project=.*&fai=23/);
$('search').value='FAI24';$('search').oninput();assert.equal($('items').children.length,1);
await $('items').children[0].onclick();assert.equal($('fai-title').textContent,'FAI 24');
assert.match($('source-image').src,/pages\/34.png$/);
$('search').value='';$('search').oninput();$('filter').value='conflict';$('filter').onchange();
assert.ok($('items').children.length>0);assert.match($('items').textContent,/FAI 54/);
$('view-3d').onclick();assert.equal($('three-area').hidden,false);
assert.equal($('geometry-empty').hidden,false);assert.match($('geometry-reason').textContent,/X_T/);
$('plan-tab').onclick();assert.equal($('plan-panel').hidden,false);
console.log('PASS: project load, 82-item index, FAI23/24 navigation, source/dependency links, conflict filter, WebGL unavailable state and plan tab.');
await fs.rm(temp,{recursive:true,force:true});dom.window.close();process.exit(0);
