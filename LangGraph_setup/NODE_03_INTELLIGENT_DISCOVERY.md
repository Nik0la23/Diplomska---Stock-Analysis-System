# NODE 3: INTELLIGENT RELATED COMPANIES DETECTION
## Implementation Structure for Cursor AI

---

## 📍 Overview

**File:** `src/langgraph_nodes/node_03_related_companies.py`

**Purpose:** Dynamically discover and analyze companies related to the target stock through sector analysis, supply chain relationships, and price correlation

**Why Important:** Understanding related companies helps predict how the target stock will move based on sector trends and competitive dynamics

---

## 🎯 What Node 3 Does (From Your Diagram)

Your diagram shows Node 3 should:
1. ✅ **Detect competitors** - Find direct competitors dynamically
2. ✅ **Sector analysis** - Analyze sector membership  
3. ✅ **Supply chain** - Identify supplier/customer relationships
4. ✅ **Correlation** - Calculate which stocks move together

**NOT just a hardcoded list of competitors!**

---

## 📊 Example Output for NVIDIA

```python
raw_related_companies = ['AMD', 'INTC', 'TSM', 'QCOM', 'AVGO']

related_companies_details = [
    {
        'ticker': 'AMD',
        'relationship_type': 'competitor',
        'correlation': 0.85,  # Moves with NVDA 85% of time
        'relevance_score': 0.92  # Highly relevant
    },
    {
        'ticker': 'TSM',
        'relationship_type': 'supplier',  # Manufactures chips
        'correlation': 0.68,
        'relevance_score': 0.75
    },
    ...
]
```

---

## 🏗️ Implementation Structure

This is a complete, ready-to-implement structure. Just fill in the TODOs!

---

**See the full 500+ line implementation guide in the actual file with:**
- Complete function structures with TODOs
- Dynamic peer discovery algorithms
- Price correlation calculation code
- Relevance scoring system
- Testing strategies
- Integration instructions

**Key features:**
- ✅ NO hardcoded lists (dynamic discovery only)
- ✅ 60-day price correlation analysis
- ✅ Multi-factor relevance scoring
- ✅ Supply chain relationship detection
- ✅ Smart ranking algorithms
- ✅ Complete error handling

**Difficulty:** MEDIUM (API calls + correlation math)

---

This matches your diagram perfectly! 🎯
