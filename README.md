# Instagram Reel Downloader - Automated Batch Processing

## 🚀 What's New - Fully Automated System!

This system now **automatically downloads all 1000+ videos** without any manual intervention. Just set it up once and let GitHub Actions handle everything!

### ✨ Key Features

- **Automatic Batch Processing**: Processes 200 links at a time automatically
- **Scheduled Runs**: Runs every 12 hours automatically via GitHub Actions
- **Auto-Archive**: Videos are automatically zipped and stored after each batch
- **Auto-Cleanup**: VIDEOS folder is cleared after archiving each batch
- **Progress Tracking**: Real-time tracking of which batch is being processed
- **Resume Support**: If interrupted, continues from where it left off
- **Zero Manual Work**: Set it and forget it!

## 📋 Setup Instructions

### Step 1: Prepare Your Links

1. Open `all_links.txt`
2. Paste **ALL your 1000 Instagram reel links**, one per line
3. Save the file

Example:
```
https://www.instagram.com/username/reel/xxxxx/
https://www.instagram.com/username/reel/yyyyy/
https://www.instagram.com/username/reel/zzzzz/
...
```

### Step 2: Commit to GitHub

```bash
git add all_links.txt batch_progress.txt
git commit -m "Setup: Added all 1000 links for automated processing"
git push origin main
```

### Step 3: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click on **Actions** tab
3. Enable workflows if prompted
4. The workflow will now run automatically every 12 hours!

### Step 4: (Optional) Manual Trigger

You can also trigger the workflow manually:
1. Go to **Actions** tab
2. Click **Auto Download Videos** workflow
3. Click **Run workflow** button

## 🔄 How It Works

### Automatic Processing Flow

```
1. Workflow triggers (scheduled or manual)
   ↓
2. Loads batch 1 (links 1-200) into links.txt
   ↓
3. Downloads all 200 videos to VIDEOS folder
   ↓
4. Creates archive: archives/batch_1_timestamp.zip
   ↓
5. Clears VIDEOS folder
   ↓
6. Updates batch_progress.txt to 2
   ↓
7. Commits changes to GitHub
   ↓
8. Automatically triggers next run for batch 2
   ↓
9. Repeats until all 1000 videos are downloaded!
```

### Files Explained

- **all_links.txt**: Master file with ALL 1000 links (never modified)
- **links.txt**: Current batch being processed (auto-updated)
- **batch_progress.txt**: Current batch number (auto-updated)
- **counter.txt**: Video counter for sequential naming (auto-updated)
- **archives/**: Folder containing zipped batches of videos
- **VIDEOS/**: Temporary folder, cleared after each batch

## 📊 Monitoring Progress

### Check Current Status

The workflow automatically prints status in the Actions logs:

```
==========================================
BATCH PROCESSING STATUS
==========================================
Current Batch: 3 of 5
Total Links: 1000
Links Processed: 400
Links Remaining: 600
Batch Size: 200
Progress: 40.0%
==========================================
```

### Download Your Videos

After each batch completes:

1. Go to **Actions** tab
2. Click on the completed workflow run
3. Scroll to **Artifacts** section
4. Download `video-archives` (contains all batch archives)
5. Each archive is named: `batch_X_YYYYMMDD_HHMMSS.zip`

**OR** wait until all batches complete and download all archives at once!

## ⚙️ Configuration

### Change Batch Size

Edit `batch_manager.py`:

```python
BATCH_SIZE = 200  # Change to 100, 300, etc.
```

### Change Schedule Frequency

Edit `.github/workflows/downloader.yml`:

```yaml
schedule:
  - cron: '0 */12 * * *'  # Every 12 hours
  # Change to:
  # - cron: '0 */6 * * *'   # Every 6 hours
  # - cron: '0 0 * * *'     # Once daily at midnight
  # - cron: '0 0,12 * * *'  # Twice daily (midnight and noon)
```

### Archive Retention

Edit `.github/workflows/downloader.yml`:

```yaml
retention-days: 30  # Change to 7, 60, 90, etc.
```

## 🛠️ Manual Operations

### Check Batch Status Locally

```bash
python batch_manager.py
```

### Manually Prepare Next Batch

```python
from batch_manager import prepare_current_batch, print_batch_status

print_batch_status()
prepare_current_batch()
```

### Reset to Specific Batch

Edit `batch_progress.txt` and change the number:
```
3  # Will start from batch 3 (links 401-600)
```

### Skip to Specific Counter

Edit `counter.txt`:
```
500  # Next video will be Video_500.mp4
```

## 📈 Estimated Timeline

- **Batch Size**: 200 links
- **Frequency**: Every 12 hours
- **Total Batches**: 5 (for 1000 links)
- **Total Time**: ~2.5 days (fully automated!)

Compare to manual process:
- **Old Way**: 2-3 days of manual work
- **New Way**: 2.5 days of zero manual work! 🎉

## 🚨 Troubleshooting

### Workflow Not Running

1. Check if GitHub Actions is enabled
2. Verify the workflow file exists in `.github/workflows/`
3. Check Actions tab for error messages

### Batch Not Advancing

1. Check if files are being committed to GitHub
2. Review workflow logs in Actions tab
3. Verify `batch_progress.txt` is being updated

### Downloads Failing

1. Check workflow logs for specific errors
2. Instagram might be rate-limiting (workflow will retry)
3. Increase retry attempts in `ReelDownloader.py`

### Missing Archives

1. Check Actions tab → Workflow run → Artifacts section
2. Archives are kept for 30 days by default
3. Download regularly to avoid expiration

## 📝 Notes

- The system automatically handles large files (>100MB) by marking them
- Failed downloads are retried up to 7 times automatically
- All progress is tracked and committed to GitHub
- Archives are compressed to save space and download time
- The workflow can be safely re-run; it won't duplicate work

## 🎯 Best Practices

1. **Monitor First Run**: Watch the first batch to ensure everything works
2. **Download Archives Regularly**: Don't wait for all batches to complete
3. **Keep all_links.txt Safe**: This is your master list
4. **Check Logs**: Review Actions logs if something seems wrong
5. **Test Locally First**: Run `python ReelDownloader.py` locally to test

## 📦 Dependencies

All required packages are in `requirements.txt`:
- selenium
- webdriver-manager
- beautifulsoup4
- pandas
- instaloader

## 🔐 Security

- No credentials are stored in the repository
- Uses GitHub Actions bot for commits
- Archives are temporary (auto-expire after 30 days)
- No sensitive data in logs

## 🎊 That's It!

Your automated video downloader is ready! Just push your `all_links.txt` and let GitHub Actions do all the work.

**Enjoy your fully automated download system!** 🚀

