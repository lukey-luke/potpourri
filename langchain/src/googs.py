from googleapiclient.discovery import build

def google_search(query, api_key, cse_id, num=10):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
    return res['items']

