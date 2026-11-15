"""Safe, bounded, read-only binary inventory. No execution of inspected binaries."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

EXTENSIONS = {'.so','.dll','.dylib','.a','.lib','.o','.obj','.cubin','.fatbin','.ptx','.hsaco','.bc','.pyd'}
ASCII = re.compile(rb'[ -~]{7,160}')
HINTS = re.compile(rb'cudnn|cublas|gemm|conv|fusion|fused|epilogue|attention|kernel|dispatch|triton|ptx|sm_[0-9]+|gfx[0-9]+', re.I)

def binary_kind(head: bytes):
    if head.startswith(b'\x7fELF'): return 'ELF'
    if head.startswith(b'MZ'): return 'PE'
    if head[:4] in (b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe'): return 'Mach-O'
    if head.startswith(b'!<arch>\n'): return 'archive'
    if head.startswith(b'BC\xc0\xde'): return 'LLVM-bitcode'
    if head.lstrip().startswith((b'.version',b'//')) and b'.target' in head: return 'PTX-text'
    return 'unknown'

def inspect_binary(path: Path, root: Path, max_bytes=16_000_000, max_strings=35):
    path=Path(path)
    size=path.stat().st_size
    digest=hashlib.sha256()
    sample=bytearray()
    with path.open('rb') as f:
        while True:
            chunk=f.read(1024*1024)
            if not chunk: break
            digest.update(chunk)
            if len(sample)<max_bytes: sample.extend(chunk[:max_bytes-len(sample)])
    buf=bytes(sample)
    matches=[]
    for m in ASCII.finditer(buf):
        if HINTS.search(m.group()):
            matches.append({'offset':m.start(),'text':m.group().decode('ascii','replace')})
            if len(matches)>=max_strings: break
    symbols=[]
    if size<2_000_000_000:
        for cmd in (['nm','-D','--defined-only',str(path)], ['readelf','-Ws',str(path)]):
            if not shutil.which(cmd[0]): continue
            try:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=8,check=False,errors='replace')
                if result.returncode==0:
                    symbols=[line[:350] for line in result.stdout.splitlines() if re.search(r'cudnn|cublas|gemm|conv|fusion|kernel|attention',line,re.I)][:30]
                    break
            except (OSError,subprocess.TimeoutExpired): pass
    return {'path':path.relative_to(root).as_posix(),'sha256':digest.hexdigest(),'size_bytes':size,
            'format':binary_kind(buf[:512]),'sampled_bytes':len(buf),'sample_truncated':size>len(buf),
            'strings':matches,'symbols':symbols,'contains_ptx_markers':b'.version' in buf and b'.target' in buf,
            'contains_cuda_markers':b'__nv_relfatbin' in buf or b'__cudaFat' in buf or b'.nv_fatbin' in buf}

def scan_binaries(root: str, max_files=0):
    root=Path(root).resolve()
    if not root.exists(): return []
    records=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink() or not (p.suffix.lower() in EXTENSIONS or '.so.' in p.name): continue
        if max_files and len(records)>=max_files: break
        try: records.append(inspect_binary(p,root))
        except (OSError,ValueError) as e: records.append({'path':p.relative_to(root).as_posix(),'error':str(e)[:250]})
    return records

def binary_tasks(records):
    tasks=[]
    for r in records:
        if 'sha256' not in r: continue
        if not r['strings'] and not r['symbols']: continue
        path=r['path']; provider=path.split('/')[0] if '/' in path else 'unknown'
        ctx=json.dumps({'format':r['format'],'sha256':r['sha256'],'strings':r['strings'],'symbols':r['symbols'],
                        'contains_cuda_markers':r['contains_cuda_markers'],'sample_truncated':r['sample_truncated']},ensure_ascii=False)
        tasks.append({'task_id':hashlib.sha256(('binary:'+r['sha256']+':'+path).encode()).hexdigest()[:20],
            'repository':'binary:'+provider,'revision':r['sha256'],'source_file':path,'source_function':'UNKNOWN',
            'source_lines':'UNKNOWN','context':ctx[:16000],'line':0,'evidence_kind':'binary',
            'binary_sha256':r['sha256']})
    return tasks
