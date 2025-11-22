# ✅ Setup Checklist

## Before You Start

- [ ] You have all your Instagram reel links ready (up to 1000+)
- [ ] You have a GitHub repository set up
- [ ] You have Git installed locally
- [ ] You have Python 3.x installed (for local testing)

---

## Step 1: Prepare Files (5 minutes)

- [ ] Open `all_links.txt`
- [ ] Paste ALL your Instagram reel links (one per line)
- [ ] Save the file
- [ ] Verify `batch_progress.txt` contains: `1`
- [ ] Verify `counter.txt` contains: `1`

---

## Step 2: Test Locally (Optional but Recommended)

- [ ] Open terminal/PowerShell
- [ ] Run: `python test_setup.py --verify`
- [ ] Check that all verifications pass ✅
- [ ] Fix any issues reported

---

## Step 3: Commit to GitHub

- [ ] Run: `git add .`
- [ ] Run: `git commit -m "Setup automated batch processing"`
- [ ] Run: `git push origin main`
- [ ] Verify files are pushed to GitHub

---

## Step 4: Enable GitHub Actions

- [ ] Go to your repository on GitHub
- [ ] Click **Actions** tab
- [ ] If prompted, click "I understand my workflows, go ahead and enable them"
- [ ] Verify workflow appears: "Auto Download Videos"

---

## Step 5: Start First Run

Choose one option:

### Option A: Manual Start (Immediate)
- [ ] In Actions tab, click **Auto Download Videos**
- [ ] Click **Run workflow** dropdown
- [ ] Click green **Run workflow** button
- [ ] Wait a few seconds, refresh page
- [ ] See workflow running with yellow spinner

### Option B: Automatic Start (Wait for schedule)
- [ ] Just wait - workflow runs every 12 hours automatically
- [ ] First run will happen at next scheduled time

---

## Step 6: Monitor Progress

- [ ] Go to Actions tab
- [ ] Click on the running workflow
- [ ] Click on "download-videos" job
- [ ] Expand steps to see logs
- [ ] Look for "BATCH PROCESSING STATUS"
- [ ] Verify correct batch number and progress

---

## Step 7: Verify First Batch Complete

After first batch completes:

- [ ] Workflow shows green checkmark ✅
- [ ] Scroll to bottom of workflow run
- [ ] See "Artifacts" section
- [ ] See `video-archives` artifact listed
- [ ] Click to download (test that archives work)

---

## Step 8: Verify Automation

- [ ] Check that `batch_progress.txt` increased (2 instead of 1)
- [ ] Check that `counter.txt` increased (~201 instead of 1)
- [ ] See new commit from "github-actions[bot]"
- [ ] Verify second workflow run started automatically

---

## Step 9: Download Videos (When Ready)

After batches complete:

- [ ] Go to Actions tab
- [ ] Click each completed workflow run
- [ ] Download `video-archives` artifact
- [ ] Extract zip files
- [ ] Organize videos as needed

---

## Verification Checklist

### Files Created
- [ ] `all_links.txt` - Contains all 1000 links
- [ ] `batch_progress.txt` - Contains batch number
- [ ] `counter.txt` - Contains video counter
- [ ] `batch_manager.py` - Exists
- [ ] `test_setup.py` - Exists
- [ ] `.github/workflows/downloader.yml` - Exists
- [ ] `VIDEOS/.gitkeep` - Exists

### Files Modified
- [ ] `ReelDownloader.py` - Has batch manager import
- [ ] `README.md` - Updated with new docs
- [ ] `.gitignore` - Updated

### Configuration
- [ ] Batch size set correctly (default: 200)
- [ ] Schedule frequency set (default: every 12 hours)
- [ ] Archive retention set (default: 30 days)

---

## Troubleshooting Checklist

If something goes wrong:

### Workflow not running
- [ ] Check if Actions is enabled
- [ ] Verify workflow file syntax (use GitHub's validator)
- [ ] Check branch is `main` (not `master`)

### Batch not advancing
- [ ] Check workflow logs for errors
- [ ] Verify files are being committed
- [ ] Check git push succeeded

### Downloads failing
- [ ] Check workflow logs for specific error
- [ ] Verify links in `all_links.txt` are valid
- [ ] Check if Instagram is blocking (rate limit)

### Archives missing
- [ ] Check workflow completed successfully
- [ ] Verify step "Archive and clear VIDEOS folder" ran
- [ ] Check if any videos were downloaded
- [ ] Look in Artifacts section of workflow run

---

## Success Indicators

You'll know everything is working when:

- ✅ Workflow runs automatically every 12 hours
- ✅ Each run processes one batch (200 links)
- ✅ Archives appear in workflow artifacts
- ✅ `batch_progress.txt` increases after each run
- ✅ New commits from github-actions[bot]
- ✅ Progress visible in logs
- ✅ No manual intervention needed

---

## Timeline Expectations

For 1000 links with default settings:

- **Batch size**: 200 links
- **Total batches**: 5
- **Frequency**: Every 12 hours
- **Total time**: ~2.5 days
- **Manual work**: 5 minutes (initial setup only)

---

## What to Watch For

### First 24 Hours
- [ ] First batch completes (~12 hours)
- [ ] Archive created successfully
- [ ] Second batch starts automatically
- [ ] Second batch completes (~24 hours)

### After 48 Hours
- [ ] At least 4 batches completed
- [ ] 800+ videos processed
- [ ] Multiple archives available

### After 60 Hours
- [ ] All 5 batches completed
- [ ] All 1000 videos processed
- [ ] All archives downloaded
- [ ] Mission accomplished! 🎉

---

## Post-Completion Checklist

After all batches complete:

- [ ] Download all video archives
- [ ] Verify all videos extracted correctly
- [ ] Check total video count matches expectations
- [ ] Archive the archives (backup)
- [ ] Clean up GitHub Actions artifacts if needed
- [ ] Update `all_links.txt` for next batch of links (if any)

---

## Optional: Advanced Configuration

If you want to customize:

- [ ] Change batch size in `batch_manager.py`
- [ ] Change schedule in `downloader.yml`
- [ ] Change retention in `downloader.yml`
- [ ] Test changes locally first
- [ ] Commit and push changes
- [ ] Verify workflow updates

---

## Notes

- ✏️ Keep this checklist handy for reference
- 📝 Check off items as you complete them
- 🔄 Repeat for future batches of 1000 links
- 💾 Save your `all_links.txt` as backup
- 📊 Monitor first run closely to ensure everything works

---

**Once you've checked off all items above, your automated system is ready!**

**Estimated setup time: 5-10 minutes**  
**Estimated automation time: 2.5 days (zero manual work)**  

**Good luck! 🚀**
