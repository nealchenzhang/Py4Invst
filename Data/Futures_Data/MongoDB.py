# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:59:19 2017

@author: Neal Chen Zhang
"""
import datetime as dt
from pymongo import MongoClient
import pprint

client = MongoClient('localhost', 27017)

db = client.test_database

collection = db.test_collection
post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": dt.datetime.utcnow()}

posts = db.posts
post_id = posts.insert_one(post).inserted_id
post_id

db.collection_names(include_system_collections=False)

pprint.pprint(posts.find_one())
pprint.pprint(posts.find_one({"author": "Mike"}))
