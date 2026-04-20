Here is a clean, **production-ready README.md** with corrected URLs, proper Cypher, and structured steps.
# 📊 Neo4j Graph Database – Social Network Dataset

This guide walks through creating a **User–Post–Relationship graph** in Neo4j using CSV files.

---

## 📁 Dataset (Updated URLs)

Use **RAW GitHub URLs** (mandatory for Neo4j):

* Users

  ```
  https://raw.githubusercontent.com/mani24singh/RAG-Mastery/main/22.%20Introduction%20to%20Graph%20Databases%20and%20Cypher%20Query%20Language%20with%20Langchain/users_social.csv
  ```

* Posts

  ```
  https://raw.githubusercontent.com/mani24singh/RAG-Mastery/main/22.%20Introduction%20to%20Graph%20Databases%20and%20Cypher%20Query%20Language%20with%20Langchain/posts.csv
  ```

* Relationships

  ```
  https://raw.githubusercontent.com/mani24singh/RAG-Mastery/main/22.%20Introduction%20to%20Graph%20Databases%20and%20Cypher%20Query%20Language%20with%20Langchain/relationships.csv
  ```

---

## ⚙️ Step 0: Constraints (Recommended)

```cypher
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User)
REQUIRE u.userId IS UNIQUE;

CREATE CONSTRAINT post_id_unique IF NOT EXISTS
FOR (p:Post)
REQUIRE p.postId IS UNIQUE;
```

---

## 👤 Step 1: Create User Nodes

```cypher
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/mani24singh/RAG-Mastery/main/22.%20Introduction%20to%20Graph%20Databases%20and%20Cypher%20Query%20Language%20with%20Langchain/users_social.csv' AS row

WITH row
WHERE row.userId IS NOT NULL

MERGE (u:User {userId: toInteger(row.userId)})
SET u.name = row.name,
    u.age = toInteger(row.age),
    u.city = row.city;
```

---

## 📝 Step 2: Create Post Nodes + POSTED Relationship

```cypher
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/mani24singh/RAG-Mastery/main/22.%20Introduction%20to%20Graph%20Databases%20and%20Cypher%20Query%20Language%20with%20Langchain/posts.csv' AS row

WITH row
WHERE row.userId IS NOT NULL AND row.postId IS NOT NULL

MATCH (u:User {userId: toInteger(row.userId)})

MERGE (p:Post {postId: toInteger(row.postId)})
SET p.content = row.content,
    p.timestamp = datetime(row.timestamp)

MERGE (u)-[:POSTED]->(p);
```

---

## 🔗 Step 3: Create Relationships (FRIEND, LIKES)

```cypher
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/mani24singh/RAG-Mastery/main/22.%20Introduction%20to%20Graph%20Databases%20and%20Cypher%20Query%20Language%20with%20Langchain/relationships.csv' AS row

WITH row
WHERE row.userId1 IS NOT NULL AND row.userId2 IS NOT NULL

MATCH (u1:User {userId: toInteger(row.userId1)})
MATCH (u2:User {userId: toInteger(row.userId2)})

MERGE (u1)-[:FRIEND]->(u2)
MERGE (u1)-[:LIKES]->(u2);
```

---

## 🔍 Step 4: Query Data

### 4.1 Retrieve Posted Details

```cypher
MATCH p = ()-[:POSTED]->()
RETURN p LIMIT 25;
```

---

### 4.2 Retrieve All Users

```cypher
MATCH (u:User)
RETURN u;
```

---

### 4.3 Retrieve All Posts

```cypher
MATCH (p:Post)
RETURN p;
```

---

### 4.4 Retrieve Friends of a Specific User

```cypher
MATCH (u:User {name: 'John'})-[:FRIEND]-(f:User)
RETURN f.name;
```

---

### 4.5 Retrieve Posts Made by Friends

```cypher
MATCH (u:User {name: 'John'})-[:FRIEND]-(f:User)-[:POSTED]->(p:Post)
RETURN f.name, p.content;
```

---

### 4.6 Count Friends per User

```cypher
MATCH (u:User)-[:FRIEND]-(f:User)
RETURN u.name, COUNT(f) AS numberOfFriends
ORDER BY numberOfFriends DESC;
```

---

## ⚠️ Common Issues & Fixes

### ❌ Error: `Characters after quote`

* Cause: Using GitHub UI URL instead of RAW
* Fix: Always use `raw.githubusercontent.com`

---

### ❌ `toInteger()` / `datetime()` fails

* Ensure:

  * No null values
  * Correct formats:

    ```
    datetime → 2023-10-01T12:30:00
    ```

---

### ❌ Duplicate Data

* Use `MERGE` instead of `CREATE`
* Add constraints (Step 0)

---

## 🚀 Best Practices

* Use **MERGE for idempotent loads**
* Always apply **constraints before ingestion**
* Validate CSV in advance (no broken quotes)
* For large datasets:

  ```cypher
  USING PERIODIC COMMIT 1000
  LOAD CSV ...
  ```

