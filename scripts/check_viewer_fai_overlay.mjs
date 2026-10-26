// Isolated browser and project: leaves the engineering project's annotations untouched.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {randomUUID} from 'node:crypto';
import {createRequire} from 'node:module';
const require=createRequire('/tmp/oracle-viewer-ui-test/package.json');
const {chromium}=require('playwright');
const origin='http://127.0.0.1:7870',root='talker/state/viewer',source='9721ec1c27e045aa955028e8a2807016';
const id=randomUUID().replaceAll('-',''),dir=`${root}/${id}`,out='../reports/oracle-viewer/fai-overlay';
const data=JSON.parse(await fs.readFile(`${root}/${source}/project.json`,'utf8'));
data.id=id;data.title='FAI overlay isolated verification';data.tasks=[];data.events=[];
await fs.mkdir(dir);await fs.writeFile(`${dir}/project.json`,JSON.stringify(data));await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium-browser',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1680,height:1050}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
const snap=()=>page.evaluate(()=>window.__oracleViewerDiagnostics());
const project=async()=>await (await fetch(`${origin}/api/viewer/projects/${id}`)).json();
const waitReady=()=>page.waitForFunction(()=>window.__oracleViewerDiagnostics?.().triangles===271171,{timeout:90000});
try{
 await page.goto(`${origin}/viewer?project=${id}&render-test=1`);await waitReady();await page.locator('#top').click();await page.waitForTimeout(600);
 assert.equal((await snap()).markers.length,2);
 await page.screenshot({path:out+'/fai-overview.png'});
 await page.locator('.marker[data-fai="23"]').click();await page.waitForFunction(()=>document.querySelector('#fai-title').textContent==='FAI 23');
 assert.equal(await page.locator('#source-page').inputValue(),'33');
 await page.locator('#view-compare').click();await page.locator('#source-image').evaluate(img=>img.decode());
 assert.equal((await snap()).mode,'compare');assert.equal(await page.locator('#source-area').isVisible(),true);assert.equal(await page.locator('#three-area').isVisible(),true);
 await page.screenshot({path:out+'/fai23-compare.png'});
 await page.locator('#marker-mode').selectOption('active');await page.waitForTimeout(200);assert.deepEqual((await snap()).markers.filter(m=>m.visible).map(m=>m.fai),[23]);
 await page.locator('#marker-mode').selectOption('none');await page.waitForTimeout(200);assert.equal((await snap()).markers.some(m=>m.visible),false);
 await page.locator('#marker-mode').selectOption('all');await page.locator('#view-3d').click();
 const before=(await project()).items.find(i=>i.number===23).annotation.anchors[0].label_offset;
 const label=await page.locator('.marker[data-fai="23"]').boundingBox();assert.ok(label);
 await page.mouse.move(label.x+label.width/2,label.y+label.height/2);await page.mouse.down();await page.mouse.move(label.x+label.width/2+70,label.y+label.height/2+50,{steps:10});await page.mouse.up();
 await page.waitForTimeout(1000);
 const after=(await project()).items.find(i=>i.number===23).annotation.anchors[0].label_offset;assert.notDeepEqual(after,before);
 await page.reload();await waitReady();assert.deepEqual((await project()).items.find(i=>i.number===23).annotation.anchors[0].label_offset,after);
 await page.locator('#search').fill('FAI 25');await page.locator('#items .item').first().click();await page.locator('#top').click();await page.waitForTimeout(400);
 const bounds=await page.locator('#canvas-host canvas').boundingBox();let picked=false;
 outer:for(const y of [.14,.16,.18,.82,.86,.88,.3,.6])for(const x of [.5,.4,.6,.35,.36,.37,.38,.62,.63,.64,.65]){
 await page.mouse.click(bounds.x+bounds.width*x,bounds.y+bounds.height*y);if((await snap()).selected.length){picked=true;break outer;}}
 assert.ok(picked);assert.equal(await page.locator('#bind').isDisabled(),false);
 await page.locator('#binding-note').fill('Isolated browser annotation verification');await page.locator('#bind').click();
 await page.waitForFunction(()=>document.querySelector('#binding-status').textContent.includes('待复核'));
 assert.equal((await project()).items.find(i=>i.number===25).annotation.status,'pending_review');
 await page.locator('#review-annotation').click();await page.waitForFunction(()=>document.querySelector('#binding-status').textContent==='位置已复核');
 const saved=(await project()).items.find(i=>i.number===25);assert.equal(saved.annotation.status,'reviewed');assert.equal(saved.annotation.metrology_ready,false);assert.equal(saved.binding,null);
 await page.locator('#unbind').click();await page.waitForFunction(()=>document.querySelector('#binding-status').textContent==='尚未标记');assert.equal((await project()).items.find(i=>i.number===25).annotation,null);
 assert.deepEqual(errors,[]);await fs.writeFile(out+'/overlay-test.json',JSON.stringify({status:'passed',checks:['FAI23/24 pending candidates','MSOP page link','side-by-side compare','marker visibility','drag and reload persistence','ray-pick/save/review/remove','review does not qualify metrology'],errors},null,2));
 console.log('PASS: FAI overlay, source links, drag persistence, preview point picking, save/review/remove.');
}finally{await browser.close();await fs.rm(dir,{recursive:true,force:true});}
