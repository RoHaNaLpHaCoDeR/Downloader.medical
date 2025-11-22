# Scaling Configuration Guide

## Processing Different Link Volumes

The system automatically handles **any number of links**. Here's how to optimize for different scales:

---

## 📊 Comparison Table

| Links | Batch Size | Schedule | Total Batches | Duration | Manual Work |
|-------|-----------|----------|---------------|----------|-------------|
| 1,000 | 200 | 12h | 5 | 2.5 days | 5 min |
| 1,000 | 500 | 6h | 2 | 0.5 days | 5 min |
| 10,000 | 200 | 12h | 50 | 25 days | 5 min |
| 10,000 | 500 | 6h | 20 | 5 days | 5 min |
| 10,000 | 1000 | 3h | 10 | 1.25 days | 5 min |
| 100,000 | 500 | 6h | 200 | 50 days | 5 min |
| 100,000 | 1000 | 3h | 100 | 12.5 days | 5 min |

---

## ⚙️ Configuration Options

### 1. Batch Size (batch_manager.py)

```python
# Conservative (Safer, Slower)
BATCH_SIZE = 100  # Small batches, less GitHub stress

# Balanced (Default)
BATCH_SIZE = 200  # Good for most cases

# Moderate
BATCH_SIZE = 500  # Faster processing

# Aggressive (Fastest)
BATCH_SIZE = 1000  # Large batches, faster completion
```

**Considerations:**
- Larger batches = Fewer workflow runs = Faster
- Smaller batches = More frequent commits = Better progress tracking
- GitHub Actions free tier: 2,000 minutes/month
- Each batch takes ~30-60 minutes depending on download speed

### 2. Schedule Frequency (.github/workflows/downloader.yml)

```yaml
# Conservative (Every 24 hours)
- cron: '0 0 * * *'

# Balanced (Every 12 hours) - DEFAULT
- cron: '0 */12 * * *'

# Moderate (Every 6 hours)
- cron: '0 */6 * * *'

# Aggressive (Every 3 hours)
- cron: '0 */3 * * *'

# Maximum (Every hour) - Not recommended
- cron: '0 * * * *'
```

**Note:** More frequent = Faster completion but uses more GitHub Actions minutes

### 3. Archive Retention (.github/workflows/downloader.yml)

```yaml
# Short-term
retention-days: 7

# Medium-term (Default)
retention-days: 30

# Long-term
retention-days: 90
```

---

## 🎯 Recommended Configurations

### Small Project (1,000 - 5,000 links)
```python
# batch_manager.py
BATCH_SIZE = 200
```
```yaml
# downloader.yml
- cron: '0 */12 * * *'  # Every 12 hours
retention-days: 30
```
**Duration:** 2-12 days | **Downloads:** Regular

---

### Medium Project (5,000 - 20,000 links)
```python
# batch_manager.py
BATCH_SIZE = 500
```
```yaml
# downloader.yml
- cron: '0 */6 * * *'  # Every 6 hours
retention-days: 30
```
**Duration:** 5-20 days | **Downloads:** Every few days

---

### Large Project (20,000 - 100,000 links)
```python
# batch_manager.py
BATCH_SIZE = 1000
```
```yaml
# downloader.yml
- cron: '0 */3 * * *'  # Every 3 hours
retention-days: 14  # Download more frequently
```
**Duration:** 6-25 days | **Downloads:** Weekly

---

### Maximum Speed (Any size, fastest possible)
```python
# batch_manager.py
BATCH_SIZE = 1000
```
```yaml
# downloader.yml
- cron: '0 */3 * * *'  # Every 3 hours
retention-days: 7
```
**Warning:** Uses more GitHub Actions minutes

---

## 📈 Real-World Examples

### Example 1: 10,000 Instagram Reels

**Goal:** Process in 1 week

**Configuration:**
```python
BATCH_SIZE = 500  # 20 batches total
```
```yaml
- cron: '0 */4 * * *'  # Every 4 hours (6 batches/day)
```

**Result:**
- 20 batches ÷ 6 per day = 3.3 days
- ✅ Completes in under 4 days!

---

### Example 2: 50,000 Videos

**Goal:** Process in 2 weeks

**Configuration:**
```python
BATCH_SIZE = 1000  # 50 batches total
```
```yaml
- cron: '0 */3 * * *'  # Every 3 hours (8 batches/day)
```

**Result:**
- 50 batches ÷ 8 per day = 6.25 days
- ✅ Completes in under 1 week!

---

### Example 3: 100,000 Videos

**Goal:** Process in 1 month

**Configuration:**
```python
BATCH_SIZE = 500  # 200 batches total
```
```yaml
- cron: '0 */6 * * *'  # Every 6 hours (4 batches/day)
```

**Result:**
- 200 batches ÷ 4 per day = 50 days
- ✅ Completes in ~7 weeks

**Better option:**
```python
BATCH_SIZE = 1000  # 100 batches total
```
```yaml
- cron: '0 */3 * * *'  # Every 3 hours (8 batches/day)
```

**Result:**
- 100 batches ÷ 8 per day = 12.5 days
- ✅ Completes in under 2 weeks!

---

## 🚨 GitHub Actions Limits

### Free Tier
- **2,000 minutes/month**
- **500 MB artifact storage**

### Calculating Usage

**Minutes per batch:** ~30-60 minutes (varies by download speed)

**Example for 10,000 links:**
- 50 batches × 45 min avg = 2,250 minutes
- ⚠️ Exceeds free tier by 250 minutes

**Solutions:**
1. Increase batch size (fewer runs)
2. Use paid tier ($0.008/minute after free tier)
3. Spread across multiple months

**Optimized for 10,000 links (free tier):**
```python
BATCH_SIZE = 500  # 20 batches
# 20 × 45 min = 900 minutes ✅ Under limit!
```

### Artifact Storage

Each archive ~100-500 MB (depends on video sizes)

**Example:**
- 50 batches × 300 MB = 15 GB temporarily
- But only stored for `retention-days`
- Downloads clear storage

**Tip:** Download and delete old artifacts regularly

---

## 🔄 How to Change Configuration

### Step 1: Edit Batch Size

```bash
# Edit batch_manager.py
# Change line 7:
BATCH_SIZE = 500  # Your desired size
```

### Step 2: Edit Schedule

```bash
# Edit .github/workflows/downloader.yml
# Change line 8:
- cron: '0 */6 * * *'  # Your desired frequency
```

### Step 3: Commit and Push

```bash
git add batch_manager.py .github/workflows/downloader.yml
git commit -m "Update batch size and schedule"
git push origin main
```

### Step 4: Verify

- New settings apply to next workflow run
- Check Actions tab to confirm

---

## 📊 Monitoring Large Batches

For 10,000+ links, monitor progress:

1. **Check batch status:**
   ```bash
   python batch_manager.py
   ```

2. **GitHub Actions dashboard:**
   - Shows current batch
   - Estimated completion
   - Success/failure status

3. **Download archives regularly:**
   - Don't wait for all batches
   - Download every week to manage storage

---

## 💡 Pro Tips

### Tip 1: Start Conservative
- Begin with default settings
- Monitor first few batches
- Adjust if too slow/fast

### Tip 2: Download Regularly
- Don't wait for all batches to complete
- Download archives weekly
- Prevents artifact expiration

### Tip 3: Test Locally First
```bash
python test_setup.py --verify
```

### Tip 4: Monitor GitHub Actions Usage
- Settings → Billing → Actions minutes
- Track usage to stay within limits

### Tip 5: Pause if Needed
- Disable workflow temporarily
- Will resume from last batch when re-enabled

---

## 🎯 Quick Decision Guide

**Choose your priority:**

### ⚡ Speed Priority
- BATCH_SIZE = 1000
- cron: '0 */3 * * *'
- May exceed free tier for large volumes

### 💰 Cost Priority (Free Tier)
- BATCH_SIZE = 500
- cron: '0 */12 * * *'
- Stays within free limits

### ⚖️ Balanced
- BATCH_SIZE = 500
- cron: '0 */6 * * *'
- Good speed, reasonable cost

---

## 📋 Configuration Template

```python
# batch_manager.py
# For _____ links, optimal batch size:

# 1,000 - 5,000 links
BATCH_SIZE = 200

# 5,000 - 20,000 links
BATCH_SIZE = 500

# 20,000 - 100,000 links
BATCH_SIZE = 1000

# 100,000+ links
BATCH_SIZE = 2000  # Experimental
```

```yaml
# .github/workflows/downloader.yml
# For completion timeline:

# Slow (weeks)
- cron: '0 0 * * *'  # Daily

# Normal (days)
- cron: '0 */12 * * *'  # Twice daily

# Fast (days)
- cron: '0 */6 * * *'  # 4x daily

# Fastest (hours-days)
- cron: '0 */3 * * *'  # 8x daily
```

---

## ✅ Your Situation: 10,000 Links

**Recommended Configuration:**

```python
# batch_manager.py
BATCH_SIZE = 500  # 20 batches total
```

```yaml
# .github/workflows/downloader.yml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
retention-days: 30
```

**Result:**
- **20 batches** total
- **4 batches/day**
- **5 days** total duration
- **~900 minutes** GitHub Actions usage (within free tier!)
- **Still only 5 minutes of your time** for initial setup

**No code changes needed** - the system automatically handles any number of links! Just update the configuration values above for optimal performance.
