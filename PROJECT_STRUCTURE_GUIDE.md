# 🏠 Project Structure Guide - Read This When You Feel Lost!

> **Simple Rule:** Each folder has ONE job. Each file does ONE thing.

---

## 📂 The House Layout (File Structure)

```
Diplomska/                          👈 THE HOUSE (your entire project)
│
├── .env.example                    👈 🔑 SECRET KEYS TEMPLATE
├── .gitignore                      👈 🚫 KEEP OUT LIST
├── requirements.txt                👈 📦 SHOPPING LIST (pip install -r requirements.txt)
├── README.md                       👈 📖 HOUSE MANUAL
│
├── data/                           👈 📁 THE STORAGE ROOM
│   ├── stock_prices.db             👈 💾 THE FILING CABINET (SQLite database)
│   └── cache/                      👈 🗄️  QUICK ACCESS DRAWER
│
├── scripts/                        👈 🔧 THE TOOLBOX
│   └── setup_database.py           👈 🛠️  "Build the Database" tool
│
├── src/                            👈 🏭 THE FACTORY (where work happens)
│   │
│   ├── database/                   👈 💾 DATABASE DEPARTMENT
│   │   ├── schema.sql              👈 📐 Database blueprint
│   │   └── db_manager.py           👈 📋 DATABASE CLERK (ONLY file that talks to DB!)
│   │
│   ├── utils/                      👈 🧰 HELPER TOOLS
│   │   ├── config.py               👈 ⚙️  Reads .env file
│   │   ├── logger.py               👈 📝 Writes logs
│   │   └── helpers.py              👈 🔨 Small utilities
│   │
│   ├── langgraph_nodes/            👈 🤖 THE 16 AI WORKERS
│   │   ├── node_01_*.py            👈    Worker #1: Fetch prices
│   │   ├── node_02_*.py            👈    Worker #2: Calculate indicators
│   │   ├── node_08_*.py            👈    Worker #8: NEWS LEARNING (thesis innovation!)
│   │   └── node_16_*.py            👈    Worker #16: Final decision
│   │
│   ├── graph/                      👈 🗺️  WORKFLOW MANAGER
│   │   └── graph_builder.py        👈    Connects the 16 workers
│   │
│   └── visualization/              👈 📊 CHART MAKER
│       └── plots.py                👈    Makes pretty graphs
│
├── streamlit_app/                  👈 🖥️  THE DASHBOARD (what users see)
│   ├── main.py                     👈 🏠 HOME PAGE
│   ├── tabs/                       👈 📑 DIFFERENT SCREENS
│   └── components/                 👈 🧩 REUSABLE UI PIECES
│
├── tests/                          👈 🧪 THE TESTING LAB
│   └── test_nodes/                 👈 🔬 Test each worker
│
└── .cursor/rules/                  👈 📜 CODING RULES (AI assistant guidelines)
```

---

## 🔄 The Data Flow (How Everything Connects)

### **Think of it like a factory assembly line:**

```
1. USER CLICKS BUTTON
   ↓
2. STREAMLIT DASHBOARD (streamlit_app/main.py)
   "Someone wants to analyze AAPL stock!"
   ↓
3. GRAPH MANAGER (src/graph/graph_builder.py)
   "Okay, I'll send it through all 16 workers!"
   ↓
4. NODE 1 → NODE 2 → NODE 3 → ... → NODE 16
   🤖 Fetch  🤖 Tech   🤖 News      🤖 Final
   price     analysis                signal
   ↓         ↓         ↓             ↓
   ALL SAVE TO DATABASE (via db_manager.py)
   ↓
5. DASHBOARD SHOWS RESULTS
   📊 Charts, tables, recommendations
```

---

## 🎯 Where to Put New Things

| I'm building... | Put it in... | Example filename |
|----------------|--------------|------------------|
| A new node | `src/langgraph_nodes/` | `node_01_price_fetch.py` |
| Database function | `src/database/db_manager.py` | Add new function here |
| Helper function | `src/utils/helpers.py` | Add new function here |
| The graph | `src/graph/` | `graph_builder.py` |
| Dashboard page | `streamlit_app/` | `main.py` |
| A test | `tests/test_nodes/` | `test_node_01.py` |

---

## 🚦 The Golden Rules

### **Rule 1: Never Talk to Database Directly**

❌ **WRONG:**
```python
import sqlite3
conn = sqlite3.connect("data/stock_prices.db")
```

✅ **CORRECT:**
```python
from src.database.db_manager import cache_price_data
cache_price_data(ticker, df)  # Let the clerk handle it!
```

### **Rule 2: Always Get Settings from config.py**

❌ **WRONG:**
```python
api_key = "sk-123456"  # Hardcoded!
```

✅ **CORRECT:**
```python
from src.utils.config import FINNHUB_API_KEY
```

### **Rule 3: Each Node is Independent**

Nodes should NOT import other nodes:

❌ **WRONG:**
```python
from src.langgraph_nodes.node_01 import fetch_price  # Node calling another node!
```

✅ **CORRECT:**
```python
# Nodes communicate through STATE (the shared whiteboard)
def node_02(state: State) -> dict:
    price_data = state["price_data"]  # Read from shared state
    # Do work...
    return {"indicators": results}  # Write to shared state
```

### **Rule 4: Use the Logger**

✅ **ALWAYS:**
```python
from src.utils.logger import get_node_logger
logger = get_node_logger("node_01")
logger.info("Fetching price for AAPL")
```

---

## 🧩 How Imports Work (File Talks to File)

```python
# You're in: src/langgraph_nodes/node_01_price_fetch.py

from src.database.db_manager import cache_price_data
#    ^         ^              ^
#    |         |              └─ The function you want
#    |         └─ The file (clerk)
#    └─ The folder (department)

from src.utils.config import FINNHUB_API_KEY
#    ^      ^       ^      ^
#    |      |       |      └─ The variable you want
#    |      |       └─ The file
#    |      └─ The folder
#    └─ Start from src/
```

**Read imports from LEFT to RIGHT:**
- "From the utils room, in the config file, get me the API key"

---

## 📝 Building Checklist

When building Node 1, you need:

- [ ] Create file: `src/langgraph_nodes/node_01_price_fetch.py`
- [ ] Import config: `from src.utils.config import FINNHUB_API_KEY`
- [ ] Import logger: `from src.utils.logger import get_node_logger`
- [ ] Import db: `from src.database.db_manager import cache_price_data`
- [ ] Write function: `def node_01_price_fetch(state: State) -> dict:`
- [ ] Save to database: `cache_price_data(ticker, data)`
- [ ] Return state update: `return {"price_data": data}`
- [ ] Create test: `tests/test_nodes/test_node_01.py`

---

## 🆘 Common Questions

### Q: "Where do I save data?"
**A:** Always through `db_manager.py`. Never directly to SQLite!

### Q: "How do nodes talk to each other?"
**A:** Through **State** (the shared whiteboard). Node 1 writes, Node 2 reads.

### Q: "Where do I put API keys?"
**A:** In `.env` file (copy from `.env.example`). Read via `config.py`.

### Q: "I'm building Node 8, where does it go?"
**A:** `src/langgraph_nodes/node_08_news_learning.py`

### Q: "How do I run the dashboard?"
**A:** `source venv/bin/activate && streamlit run streamlit_app/main.py`

---

## 🎯 Quick Mental Model

```
DATABASE = Filing Cabinet (1 shared storage for everyone)
    ↕️
db_manager.py = Clerk (only person who opens the cabinet)
    ↕️
NODES = 16 Workers (each does 1 specialized task)
    ↕️
GRAPH = Manager (tells workers what order to work in)
    ↕️
STREAMLIT = Display Window (shows results to you)
```

**Everyone reads from and writes to the same filing cabinet!**

---

## 📌 Remember:

1. **One folder = One purpose**
2. **One file = One responsibility**
3. **All data goes through `db_manager.py`**
4. **Nodes communicate via State, not by calling each other**
5. **Config, logging, and helpers are in `utils/`**

---

## 🚀 You're Ready When:

- ✅ You understand: Nodes → db_manager → Database
- ✅ You know where to put new node files
- ✅ You can explain: "Why do we use db_manager instead of sqlite3 directly?"
  - Answer: "So all database code is in ONE place, easier to maintain!"

---

**Now go build! Check `NODE_BUILD_GUIDE.md` for what to build.** 💪
