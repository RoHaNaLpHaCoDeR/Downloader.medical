# 🚀 Quick Start Guide

## Complete Setup in 5 Minutes

### 1. Add Your Links (2 minutes)

Open `all_links.txt` and paste all your Instagram reel links:

```
https://www.instagram.com/life_withmedicals/reel/C9oQ6NgJZqz/
https://www.instagram.com/life_withmedicals/reel/C9o0kcdpX_g/
https://www.instagram.com/life_withmedicals/reel/C9o3BXUpLH3/
... (add all 1000 links)
```

### 2. Push to GitHub (1 minute)

```bash
git add .
git commit -m "Setup automated batch processing"
git push origin main
```

### 3. Enable GitHub Actions (1 minute)

1. Go to your repo on GitHub
2. Click **Actions** tab
3. Click **"I understand my workflows, go ahead and enable them"** if prompted

### 4. Start First Run (30 seconds)

**Option A: Manual Start**
1. In Actions tab, click **Auto Download Videos**
2. Click **Run workflow** → **Run workflow**

**Option B: Automatic Start**
- Just wait! It will auto-run every 12 hours

### 5. Monitor Progress (ongoing)

Check progress anytime:
1. Go to **Actions** tab
2. Click on running/completed workflow
3. Expand steps to see logs
4. Look for "BATCH PROCESSING STATUS"

### 6. Download Your Videos

After each batch completes:
1. Go to completed workflow run
2. Scroll to **Artifacts** section at bottom
3. Click **video-archives** to download
4. Extract the zip files

---

## What Happens Automatically

✅ Every 12 hours:
- Downloads next batch of 200 videos
- Archives them as a zip file
- Clears VIDEOS folder
- Moves to next batch
- Commits progress to GitHub
- Triggers next run

✅ You get:
- All 1000 videos downloaded automatically
- No manual syncing needed
- No manual batch management
- Progress saved in case of interruptions

---

## Customization (Optional)

### Download More Frequently

Edit `.github/workflows/downloader.yml`, line 8:

```yaml
- cron: '0 */6 * * *'  # Every 6 hours instead of 12
```

### Smaller Batches

Edit `batch_manager.py`, line 7:

```python
BATCH_SIZE = 100  # Instead of 200
```

### Longer Archive Retention

Edit `.github/workflows/downloader.yml`, line 135:

```yaml
retention-days: 60  # Instead of 30
```

---

## FAQ

**Q: How long will it take to download all 1000 videos?**
A: About 2.5 days with 200 per batch, every 12 hours = 5 batches total

**Q: Can I speed it up?**
A: Yes! Change cron to `*/6` for every 6 hours = done in ~1.5 days

**Q: What if it stops in the middle?**
A: No problem! It will resume from the last completed batch automatically

**Q: Do I need to keep my computer on?**
A: No! Everything runs on GitHub's servers

**Q: How much does this cost?**
A: Free! GitHub Actions provides 2000 free minutes/month

**Q: Can I download the videos before all batches complete?**
A: Yes! Download artifacts from each completed workflow run

---

## Verification Checklist

Before you start, verify:

- [ ] `all_links.txt` contains all your links
- [ ] `batch_progress.txt` contains: `1`
- [ ] `counter.txt` contains: `1`
- [ ] Files are committed and pushed to GitHub
- [ ] GitHub Actions is enabled
- [ ] Workflow file exists: `.github/workflows/downloader.yml`

---

## Need Help?

Check the logs in GitHub Actions for detailed information about what's happening at each step.

**Everything is automated now - Enjoy!** 🎉
