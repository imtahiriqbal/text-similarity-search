# Text Similarity Search

### Dependencies
<table>

<tr> 
    <td><b>Tool</td>
    <td><b>Version</td>
</tr>

<tr>
<td>Python</td>
<td>3.11</td>
</tr>

<tr>
<td>OpenAI</td>
<td>0.28.1</td>
</tr>

<tr>
<td>Faiss (Facebook AI Similarity Search)</td>
<td>1.9.0</td>
</tr>

<tr>
<td>Numpy</td>
<td>1.26.4</td>
</tr>

</table>

### Create virtual environment
- ```python3 -m venv [ENV_NAME]```
- ```source [ENV_NAME]/bin/activate```
- Install dependencies: ```pip install -r requirements.txt```

### Create .env to store OpenAI Secrets
- ```API_KEY=[YOUR-API-KEY]```


### How it works

Input:
```
I love programming.
Machine learning is fascinating.
Deep learning is a subset of machine learning.
I enjoy writing code in Python.
```

Query:
```
I'm interested in neural networks.
```

Output:
```
🔍 Most Similar Sentences with Distance:

- Machine learning is fascinating. (L2 distance: 0.2744)
Similarity Score: 0.8628

- Deep learning is a subset of machine learning. (L2 distance: 0.3455)
Similarity Score: 0.8273
```
