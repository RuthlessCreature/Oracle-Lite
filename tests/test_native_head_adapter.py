"""Exercise the production C++ adapter reader and math, including corrupted artifacts."""
import shutil,struct,subprocess
from pathlib import Path
import numpy as np
import pytest
HEADER=Path('/home/cyg/Strata/include/strata/core/oracle_head_adapter.hpp')
@pytest.fixture(scope='module')
def executable(tmp_path_factory):
 if not HEADER.exists() or not shutil.which('g++'):pytest.skip('Strata C++ toolchain unavailable')
 d=tmp_path_factory.mktemp('native-head');s=d/'test.cpp';s.write_text('''#include "strata/core/oracle_head_adapter.hpp"
#include <cstdio>
int main(int argc,char**argv){try {strata::core::OracleHeadAdapter a;a.load(argv[1],3,5);float x[]={2,3,4},b[]={0,0,0,0,0};auto tok=a.apply(x,b);printf("%lld %.9g %.9g\\n",(long long)tok,b[1],b[3]);return 0;}catch(const std::exception&e){return 1;}}''')
 exe=d/'test';subprocess.run(['g++','-std=c++17','-O2','-I'+str(HEADER.parents[2]),str(s),'-o',str(exe)],check=True,capture_output=True);return exe

def artifact(path,ids=(1,3),scale=(1,1,1)):
 b=struct.pack('<IIIIIf',0x3141484f,3,1,2,5,2)
 for a,dtype in [(ids,'<u4'),([0,0,0],'<f4'),(scale,'<f4'),([1,1,1],'<f4'),([1,-1],'<f4'),([0,0],'<f4')]:b+=np.array(a,dtype=dtype).tobytes()
 path.write_bytes(b)

def test_native_math_and_clip(executable,tmp_path):
 p=tmp_path/'a';artifact(p);r=subprocess.run([str(executable),str(p)],capture_output=True,text=True,check=True);assert r.stdout.split()==['1','2','-2']
@pytest.mark.parametrize('corruption',['truncated','extra','duplicate','range','scale','nan'])
def test_corrupt_artifacts_rejected(executable,tmp_path,corruption):
 p=tmp_path/'a';artifact(p,ids=(1,1) if corruption=='duplicate' else (1,5) if corruption=='range' else (1,3),scale=(0,1,1) if corruption=='scale' else (float('nan'),1,1) if corruption=='nan' else (1,1,1))
 if corruption=='truncated':p.write_bytes(p.read_bytes()[:-1])
 if corruption=='extra':p.write_bytes(p.read_bytes()+b'!')
 assert subprocess.run([str(executable),str(p)],capture_output=True).returncode!=0
