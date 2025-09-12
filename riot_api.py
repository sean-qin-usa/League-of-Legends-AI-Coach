import os, asyncio, aiohttp, time, random
from typing import List, Dict, Any, Tuple

def platform_to_routing(platform: str) -> str:
    m = {
        "na1": "americas","br1":"americas","la1":"americas","la2":"americas",
        "euw1":"europe","eun1":"europe","tr1":"europe","ru":"europe",
        "kr":"asia","jp1":"asia","oc1":"sea","ph2":"sea","sg2":"sea","th2":"sea","tw2":"sea","vn2":"sea"
    }
    return m.get(platform.lower(), "americas")

def _headers():
    key = os.getenv("RIOT_API_KEY")
    if not key or key.startswith("REPLACE_ME"):
        raise RuntimeError("RIOT_API_KEY not set")
    return {"X-Riot-Token": key}

async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any]=None, label: str="req", retries: int=6) -> Any:
    params = params or {}
    for i in range(retries):
        try:
            async with session.get(url, params=params) as r:
                if r.status == 200:
                    return await r.json()
                if r.status == 429:
                    ra = float(r.headers.get("Retry-After") or "1.2")
                    await asyncio.sleep(ra + random.uniform(0.1,0.3))
                    continue
                if 500 <= r.status < 600:
                    await asyncio.sleep(0.5 + 0.2*i); continue
                txt = await r.text()
                raise RuntimeError(f"{label} failed {r.status}: {txt[:200]} url={url}")
        except aiohttp.ClientConnectorCertificateError:
            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(headers=_headers(), connector=conn) as s2:
                async with s2.get(url, params=params) as r2:
                    if r2.status == 200:
                        return await r2.json()
                    if r2.status == 429:
                        ra = float(r2.headers.get("Retry-After") or "1.2")
                        await asyncio.sleep(ra + random.uniform(0.1,0.3))
                        continue
                    txt = await r2.text()
                    raise RuntimeError(f"{label} ssl=False failed {r2.status}: {txt[:200]} url={url}")
        except Exception:
            if i == retries-1:
                raise
            await asyncio.sleep(0.3 + 0.2*i)
    raise RuntimeError(f"{label} failed after {retries} retries: {url}")

async def seed_puuids(platform: str, queue: str, min_tier: str, max_tier: str, divisions: List[str], pages: int, limit: int=200) -> List[str]:
    base = f"https://{platform}.api.riotgames.com"
    tiers = ["IRON","BRONZE","SILVER","GOLD","PLATINUM","EMERALD","DIAMOND","MASTER","GRANDMASTER","CHALLENGER"]
    lo = tiers.index(min_tier.upper())
    hi = tiers.index(max_tier.upper())
    out = []
    async with aiohttp.ClientSession(headers=_headers()) as s:
        for idx in range(lo, hi+1):
            t = tiers[idx]
            if t in ("MASTER","GRANDMASTER","CHALLENGER"):
                url = f"{base}/lol/league/v4/{t.lower()}leagues/by-queue/{queue}"
                try:
                    j = await _get_json(s, url, label="league")
                    entries = j.get("entries") if isinstance(j, dict) else j
                except Exception:
                    entries = []
            else:
                entries = []
                for d in divisions:
                    for p in range(1, pages+1):
                        url = f"{base}/lol/league/v4/entries/{queue}/{t}/{d}"
                        try:
                            chunk = await _get_json(s, url, params={"page": p}, label="league-entries")
                            if not isinstance(chunk, list) or not chunk:
                                break
                            entries.extend(chunk)
                        except Exception:
                            break
            for e in entries:
                if "puuid" in e and e["puuid"]:
                    out.append(e["puuid"])
                else:
                    sid = e.get("summonerId")
                    if not sid: 
                        continue
                    url = f"{base}/lol/summoner/v4/summoners/{sid}"
                    try:
                        summ = await _get_json(s, url, label="summoner")
                        pu = summ.get("puuid")
                        if pu: out.append(pu)
                    except Exception:
                        continue
                if len(out) >= limit:
                    return list(dict.fromkeys(out))[:limit]
    return list(dict.fromkeys(out))[:limit]

async def fetch_many_ids(routing: str, puuids: List[str], per_puuid: int=20, concurrency: int=6, queue_id: int=None, start_time=None, end_time=None) -> List[str]:
    base = f"https://{routing}.api.riotgames.com"
    sem = asyncio.Semaphore(concurrency)
    ids = []
    async with aiohttp.ClientSession(headers=_headers()) as s:
        async def _one(pu):
            # paginate by 'start' in chunks of up to 100 (API limit)
            fetched = 0
            start = 0
            while fetched < per_puuid:
                batch = min(100, per_puuid - fetched)
                params = {"count": batch, "start": start}
                if queue_id is not None: params["queue"] = int(queue_id)
                if start_time: params["startTime"] = int(start_time)
                if end_time: params["endTime"] = int(end_time)
                url = f"{base}/lol/match/v5/matches/by-puuid/{pu}/ids"
                async with sem:
                    try:
                        j = await _get_json(s, url, params=params, label="ids")
                        if not isinstance(j, list) or not j:
                            break
                        ids.extend(j); n = len(j)
                        fetched += n; start += n
                        if n < batch:
                            break
                    except Exception as e:
                        print("[ids] fail:", e)
                        break
        await asyncio.gather(*[_one(pu) for pu in puuids])
    seen = set(); out = []
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

async def fetch_matches_and_timelines(routing: str, match_ids: List[str], concurrency: int=8):
    base = f"https://{routing}.api.riotgames.com"
    sem = asyncio.Semaphore(concurrency)
    matches, timelines = {}, {}
    async with aiohttp.ClientSession(headers=_headers()) as s:
        async def _one(mid):
            async with sem:
                mu = f"{base}/lol/match/v5/matches/{mid}"
                tu = f"{base}/lol/match/v5/matches/{mid}/timeline"
                try:
                    m = await _get_json(s, mu, label="match")
                    t = await _get_json(s, tu, label="timeline")
                    matches[mid] = m if isinstance(m, dict) else {}
                    timelines[mid] = t if isinstance(t, dict) else {}
                except Exception as e:
                    print("[fetch] fail:", mid, e)
        await asyncio.gather(*[_one(mid) for mid in match_ids])
    return matches, timelines
