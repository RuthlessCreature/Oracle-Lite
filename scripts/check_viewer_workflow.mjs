import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire('/tmp/oracle-viewer-ui-test/package.json'),{chromium}=require('playwright');
const out='../reports/oracle-viewer/workflow';await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium-browser',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1680,height:1050}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
const pid='9721ec1c27e045aa955028e8a2807016',origin='http://127.0.0.1:7870';
try{
 await page.goto(`${origin}/viewer?project=${pid}&render-test=1`);
 await page.waitForFunction(()=>window.__oracleViewerDiagnostics?.().triangles>0,{timeout:90000});
 await page.locator('#open-workflow').click();await page.waitForSelector('#wf-steps button');
 await page.waitForFunction(()=>document.querySelector('#wf-content')?.textContent.includes('82 项'));
 assert.equal(await page.locator('#wf-steps button').count(),6);
 assert.equal(await page.locator('#wf-content tbody tr').count(),82);
 await page.screenshot({path:out+'/01-measurement-tasks.png'});
 for(let i=1;i<6;i++){
  await page.locator(`#wf-steps button[data-step="${i}"]`).click();
  await page.screenshot({path:out+`/0${i+1}-workflow.png`});
  if(i===2){assert.equal(await page.locator('#wf-content tbody tr').count(),7);assert.equal(await page.locator('#wf-content a[href*="/library/assets/"]').count(),14);}
  if(i===3){assert.match(await page.locator('#wf-status').textContent(),/strata-native/);assert.match(await page.locator('#wf-content').textContent(),/H[0-9]+/);}
  if(i===5){assert.match(await page.locator('#wf-content').textContent(),/计量验收：未完成/);}
 }
 await page.locator('#wf-steps button[data-step="4"]').click();await page.locator('[data-head]').first().click();
 await page.waitForFunction(()=>window.__oracleViewerDiagnostics?.().planningObjectCount>0);await page.waitForTimeout(600);
 const scene=await page.evaluate(()=>window.__oracleViewerDiagnostics());assert.ok(scene.planningObjectCount>=8);
 assert.ok(scene.markers.filter(m=>m.visible).length>=1);
 await page.screenshot({path:out+'/07-model-selected-layout.png'});
 await page.locator('#clear-planning').click();
 assert.equal((await page.evaluate(()=>window.__oracleViewerDiagnostics())).planningHead,null);
 const download=await page.request.get(`${origin}/api/viewer/projects/${pid}/workflow/export?format=csv`);assert.equal(download.status(),200);assert.match(await download.text(),/TC23024/);
 const record=await (await page.request.get(`${origin}/api/viewer/projects/${pid}/workflow/export`)).json();
 assert.equal(record.model_review.backend,'strata-native');assert.ok(record.result.heads.length>=1);assert.ok(record.result.heads.some(h=>h.fai_numbers.includes(23)&&h.fai_numbers.includes(24)));
 assert.equal(record.result.metrology_accepted,false);assert.equal(record.result.ready_for_procurement,false);assert.equal(record.stale,false);
 assert.deepEqual(errors,[]);
 await fs.writeFile(out+'/workflow-test.json',JSON.stringify({status:'passed',checks:['82 source tasks','6 workflow stages','7 manufacturer models','14 local source files','Strata native actual decision','checked candidate IDs','shared layout and BOM','conditional validation status','CSV/JSON export'],model:record.model_review,head:record.result.heads[0],scene,errors},null,2));
 console.log('PASS: workflow / manufacturer library / actual Strata decision / 3D candidate layout / BOM export');
}finally{await browser.close();}
