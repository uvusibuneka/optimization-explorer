"""Ingest source drop zone, preserving repository identities."""
import json
import subprocess
from pathlib import Path
from .indexer import iter_directory_tasks, iter_zip_tasks

def revision_for(path):
    if path.is_dir() and (path/'.git').exists():
        try:
            r=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
            if r.returncode==0: return r.stdout.strip()
        except (OSError,subprocess.TimeoutExpired): pass
    return 'UNKNOWN'

def ingest(sources, store, max_files=0, max_hits=0):
    sources=Path(sources).resolve()
    manifest=sources/'repositories.json'
    repos=json.loads(manifest.read_text())['repositories'] if manifest.exists() else []
    known={r['id'] for r in repos}
    inputs=[]
    if sources.is_dir():
        for p in sorted(sources.iterdir()):
            if p.name.startswith('.') or p.name in ('README.md','repositories.json') or p.is_symlink(): continue
            if p.is_dir(): inputs.append((p.name,p,'directory'))
            elif p.suffix.lower()=='.zip': inputs.append((p.stem,p,'zip'))
    results=[]
    for name,path,kind in inputs:
        if name not in known and known: continue
        revision=revision_for(path)
        try:
            index = iter_zip_tasks if kind == 'zip' else iter_directory_tasks
            tasks = index(str(path), name, revision, max_files or 2**31-1, max_hits or 2**31-1)
            count = store.add_indexed_tasks(tasks)
            results.append({'repository':name,'path':str(path),'revision':revision,'tasks':count})
        except (OSError,ValueError,Exception) as e:
            results.append({'repository':name,'error':f'{type(e).__name__}: {e}'})
    return {'sources':results,'stats':store.stats()}

def inventory(sources):
    sources=Path(sources)
    manifest=sources/'repositories.json'
    repos=json.loads(manifest.read_text())['repositories'] if manifest.exists() else []
    return {'repositories':[{'id':r['id'],'url':r['url'],'present':(sources/r['id']).is_dir() or (sources/(r['id']+'.zip')).is_file()} for r in repos]}
