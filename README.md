# classify-question
Using Naive Bayes to classify question topics: math, financial, health, social, ...

# Setup environment
```
    virutalenv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
```

# Run the server
```
python3 main.py
```

# HTTP request sample
```
curl --location 'http://localhost:5005/questions-type' \
--header 'Content-Type: application/json' \
--data '{
    "question": "What is rectangular?"
}'
```

# Response 
```
{
    "question_type": "math"
}
```

