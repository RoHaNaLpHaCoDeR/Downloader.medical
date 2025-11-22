# Migration Guide - Old to New System

## 🔄 Transitioning from Manual to Automated System

If you've already started downloading videos with the old manual process, here's how to transition seamlessly to the new automated system.

---

## Scenario 1: Starting Fresh (Easiest)

**If you haven't downloaded any videos yet:**

1. Put all 1000 links in `all_links.txt`
2. Ensure `batch_progress.txt` = `1`
3. Ensure `counter.txt` = `1`
4. Push to GitHub and enable Actions
5. Done! ✅

---

## Scenario 2: Already Downloaded Some Videos

**If you've already downloaded 200 videos manually (videos 1-200):**

### Option A: Continue from Where You Left Off

1. **Calculate your current position:**
   - If you have Video_1.mp4 to Video_200.mp4
   - Next batch should be batch 2 (links 201-400)

2. **Update tracking files:**
   ```bash
   # Set batch to 2
   echo 2 > batch_progress.txt
   
   # Set counter to 201 (next video number)
   echo 201 > counter.txt
   ```

3. **Setup all_links.txt:**
   - Copy ALL 1000 links into `all_links.txt` (including the 200 you already processed)
   - The system will skip them automatically

4. **Push and start:**
   ```bash
   git add all_links.txt batch_progress.txt counter.txt
   git commit -m "Migrate to automated system - starting from batch 2"
   git push origin main
   ```

### Option B: Start Fresh with All Links

1. **Archive your existing videos:**
   ```bash
   # Create a backup
   mkdir -p archives
   # Compress existing videos
   Compress-Archive -Path VIDEOS\*.mp4 -DestinationPath archives\manual_batch_1.zip
   # Clear VIDEOS folder
   Remove-Item VIDEOS\*.mp4
   ```

2. **Reset counters:**
   ```bash
   echo 1 > batch_progress.txt
   echo 1 > counter.txt
   ```

3. **Setup all_links.txt:**
   - Put all 1000 links in `all_links.txt`

4. **Note:** This will re-download everything, but you'll have a clean automated process

---

## Scenario 3: Mid-Batch Processing

**If you're in the middle of downloading a batch:**

1. **Option A: Finish Current Batch Manually**
   - Complete your current batch of 200
   - Then follow Scenario 2

2. **Option B: Skip Partial Downloads**
   - Note which videos you've downloaded (e.g., Video_1 to Video_150)
   - Set `counter.txt` to next number (151)
   - Continue with automated system
   - The partial batch will have 50 videos instead of 200

---

## Scenario 4: Multiple Batches Already Done

**If you've completed multiple batches manually (e.g., 600 videos = 3 batches):**

1. **Determine your position:**
   - 600 videos = 3 batches complete
   - Next should be batch 4

2. **Update files:**
   ```bash
   echo 4 > batch_progress.txt
   echo 601 > counter.txt
   ```

3. **Setup all_links.txt:**
   - Put all 1000 links in `all_links.txt`

4. **Commit and push:**
   ```bash
   git add all_links.txt batch_progress.txt counter.txt
   git commit -m "Migrate to automated system - starting from batch 4"
   git push origin main
   ```

---

## Verification After Migration

After setup, verify everything is correct:

1. **Run locally to test:**
   ```bash
   python batch_manager.py
   ```

2. **Check output:**
   ```
   ==========================================
   BATCH PROCESSING STATUS
   ==========================================
   Current Batch: 4 of 5        # Should match your position
   Total Links: 1000
   Links Processed: 600         # Should match videos done
   Links Remaining: 400
   Batch Size: 200
   Progress: 60.0%
   ==========================================
   ```

3. **If correct:**
   - The automated system will continue from the right position

4. **If incorrect:**
   - Adjust `batch_progress.txt` and `counter.txt` accordingly

---

## Important Notes

### Counter vs Batch Relationship

- **Batch 1**: Videos 1-200 (counter starts at 1)
- **Batch 2**: Videos 201-400 (counter starts at 201)
- **Batch 3**: Videos 401-600 (counter starts at 401)
- **Batch N**: Videos [(N-1)*200 + 1] to [N*200]

**Formula to calculate counter:**
```
counter = (batch_number - 1) * 200 + 1
```

**Example:**
- Starting batch 4: counter = (4-1) * 200 + 1 = 601

### all_links.txt Structure

MUST contain links in order:
```
Link 1     # Will be Video_1
Link 2     # Will be Video_2
...
Link 200   # Will be Video_200
Link 201   # Will be Video_201 (batch 2 starts)
...
Link 1000  # Will be Video_1000
```

---

## Common Migration Issues

### Issue: Videos getting overwritten

**Cause:** Counter is set too low  
**Fix:** Set counter to (last_video_number + 1)

### Issue: Batch seems wrong

**Cause:** batch_progress.txt doesn't match actual progress  
**Fix:** Calculate correct batch number and update file

### Issue: Duplicate downloads

**Cause:** Links in `all_links.txt` don't match what was processed  
**Fix:** Ensure `all_links.txt` contains exact same links in same order

---

## Migration Checklist

Before starting automated system:

- [ ] All 1000 links are in `all_links.txt` in correct order
- [ ] `batch_progress.txt` = (batches completed + 1)
- [ ] `counter.txt` = (total videos downloaded + 1)
- [ ] Old videos are backed up if needed
- [ ] VIDEOS folder is empty or ready for next batch
- [ ] Files are committed and pushed to GitHub
- [ ] Test run locally: `python batch_manager.py`
- [ ] Verify status output is correct

---

## Example: Real Migration

**Situation:**
- Downloaded 437 videos manually
- Videos are Video_1.mp4 through Video_437.mp4
- Have 1000 total links to process

**Steps:**

1. Calculate batch position:
   - 437 videos ÷ 200 = 2.185 batches
   - Completed: 2 full batches (400 videos)
   - Currently in batch 3 with 37 videos

2. **Option A: Continue batch 3**
   ```bash
   echo 3 > batch_progress.txt
   echo 438 > counter.txt  # Next video
   ```
   - Batch 3 will have 163 more videos (200 - 37 = 163)

3. **Option B: Start fresh batch 3**
   ```bash
   # Archive partial batch 3
   mkdir temp_archive
   mv VIDEOS/Video_401.mp4 to Video_437.mp4 temp_archive/
   
   echo 3 > batch_progress.txt
   echo 401 > counter.txt  # Start of batch 3
   ```
   - Batch 3 will download all 200 videos fresh

4. Add all 1000 links to `all_links.txt`

5. Push to GitHub:
   ```bash
   git add .
   git commit -m "Migrate: Continue from batch 3, video 438"
   git push origin main
   ```

---

## Need Help?

If you're unsure about your current position:

1. Count your downloaded videos
2. Check the highest video number
3. Use the formulas above
4. Test locally before pushing

**When in doubt, it's safer to:**
- Backup existing videos
- Reset to batch 1
- Let automation handle everything fresh

Good luck with your migration! 🚀
