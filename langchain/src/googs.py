from googleapiclient.discovery import build

def google_search(query, api_key, cse_id, num=10):
    """
    Perform a Google search using the Custom Search JSON API.

    Args:
        query (str): The search query.
        api_key (str): Your Google API key.
        cse_id (str): Your Custom Search Engine ID.
        num (int): The number of search results to return.

    Returns:
        list: A list of search results.
    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
    return res['items']
