import hashlib, json, pathlib, time, urllib.request
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/raw')
root.mkdir(parents=True, exist_ok=True)
files=[]
for period in ('2025-11','2025-12','2026-01','2026-02'):
    name=f'TRXUSDT-1m-{period}.zip'
    base=f'https://data.binance.vision/data/futures/um/monthly/klines/TRXUSDT/1m/{name}'
    for suffix in ('.CHECKSUM',''):
        url=base+suffix
        target=root/(name+suffix)
        error=None
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'RouteB-reproducible-research/1.0'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    data=response.read()
                target.write_bytes(data)
                break
            except Exception as exc:
                error=repr(exc)
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f'download failed {url}: {error}')
    sidecar=(root/(name+'.CHECKSUM')).read_text('ascii').strip()
    expected=sidecar.split()[0].lower()
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:
        raise RuntimeError(f'checksum mismatch {name}: {actual} != {expected}')
    files.append({'file':name,'url':base,'sha256':actual,'bytes':(root/name).stat().st_size,
                  'checksum_sidecar':sidecar})
report={'source':'Binance Public Data USD-M monthly TRXUSDT 1m klines','files':files}
(root/'download_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))