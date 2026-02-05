# LangGraph Stock Analysis - Cursor Rules

This folder contains focused, composable rules for the thesis project.

## Rules Overview

### Core Development Rules (Always Applied)
1. **01-state-management.md** - State-first LangGraph development
2. **02-error-handling.md** - Never break the graph
3. **03-database-operations.md** - SQLite patterns
4. **04-api-caching.md** - Cache-first API strategy
5. **05-logging-standards.md** - Structured logging
6. **06-type-hints.md** - Type hints everywhere
7. **07-node-structure.md** - Standard file structure
8. **08-testing-requirements.md** - Test every node
14. **14-documentation.md** - Comprehensive docstrings
15. **15-configuration.md** - Environment variables

### Context-Specific Rules (Applied Intelligently)
9. **09-thesis-innovation-node8.md** - Node 8 learning system (PRIMARY INNOVATION)
10. **10-two-phase-anomaly.md** - Dual anomaly detection (THESIS INNOVATION)
11. **11-adaptive-weighting.md** - Backtest & weight calculation
12. **12-parallel-execution.md** - Async patterns
13. **13-performance.md** - Optimization for <5s execution

## Rule Types

- **Always Apply** - Core standards used in all files
- **Apply Intelligently** - Context-aware (globs pattern matching)
- **Apply Manually** - Reference with @rule-name when needed

## Reference Files

Instead of copying code into rules, these rules reference:
- `@NODE_BUILD_GUIDE.md` - Complete node specifications
- `@LangGraph_setup/node_08_news_verification_COMPLETE.py` - Node 8 implementation
- `@LangGraph_setup/NEWS_LEARNING_SYSTEM_GUIDE.md` - Learning system concepts
- `@LangGraph_setup/NODE_06_STRUCTURE_FOR_CURSOR.md` - Node 6 specs
- `@LangGraph_setup/NODE_07_STRUCTURE_FOR_CURSOR.md` - Node 7 specs

## Usage

Rules are automatically applied by Cursor based on:
1. `alwaysApply: true` - Every file
2. `globs: ["**pattern"]` - Files matching pattern
3. Manual mention - `@rule-name` in chat

## Best Practices

- Keep each rule focused (< 500 lines)
- Reference files instead of copying code
- Use clear examples with ✅/❌ patterns
- Update rules as project evolves
- Check rules into git for team consistency
