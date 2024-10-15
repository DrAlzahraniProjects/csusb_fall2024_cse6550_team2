import streamlit as st
from pymilvus import connections

milvus_client_utility = connections.connect(host='localhost', port='19530')
@st.cache_resource
def get_milvus_client(uri: str, token: str = None):
    return connections.connect(host='localhost', port='19530')

collection:any
def create_collection(
    milvus_client_utility, collection_name: str, dim: int, drop_old: bool = True
):
    if milvus_client_utility.has_collection(collection_name) and drop_old:
        milvus_client_utility.drop_collection(collection_name)
    if milvus_client_utility.has_collection(collection_name):
        raise RuntimeError(
            f"Collection {collection_name} already exists. Set drop_old=True to create a new one instead."
        )
    
    collection = milvus_client_utility.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="IP",
        consistency_level="Strong",
        auto_id=True,
    )
    print("Collection created")
    return collection

def get_search_results(milvus_client_utility, collection_name, query_vector, output_fields):
    search_res = collection.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=output_fields,
    )
    return search_res