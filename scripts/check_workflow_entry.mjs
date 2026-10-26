import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire('/tmp/oracle-viewer-ui-test/package.json'),{chromium}=require('playwright');
const browser=await chromium.launch({executablePath:'/usr/bin/chromium-browser',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const origin='http://192.168.3.5:7870',pid='9721ec1c27e045aa955028e8a2807016';
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 await page.route('**/mesh',route=>route.abort());
 const response=await page.goto(origin+`/viewer?project=${pid}`);
 assert.equal(response.headers()['cache-control'],'no-store');
 for(const name of ['viewer.css','viewer.js','workflow.js'])assert.ok(await page.locator(`[${name.endsWith('css')?'href':'src'}^="/viewer-static/${name}?v="]`).count());
 await page.waitForFunction(()=>document.querySelector('#projects').value);
 await page.locator('#open-workflow').click();await page.locator('#workflow-panel').waitFor({state:'visible'});
 await page.waitForFunction(()=>document.querySelector('#wf-content').textContent.includes('82 项'));
 const b=await page.locator('#workflow-panel').boundingBox();assert.ok(b.y<900&&b.height>300);
 await page.reload();await page.waitForFunction(()=>document.querySelector('#projects').value);
 await page.locator('#open-workflow').click();await page.locator('#workflow-panel').waitFor({state:'visible'});
 const fallback=await browser.newPage({viewport:{width:1440,height:900}});
 await fallback.route('**/workflow.js*',route=>route.abort());await fallback.route('**/mesh',route=>route.abort());
 await fallback.goto(origin+`/viewer?project=${pid}`);await fallback.locator('#open-workflow').click();
 await fallback.locator('#workflow-panel').waitFor({state:'visible'});
 assert.match(await fallback.locator('#wf-status').textContent(),/流程脚本尚未就绪/);
 console.log('PASS: LAN button opens visible workflow, versioned assets/no-store, reload, module failure feedback');
}finally{await browser.close();}
