"""
Test and Setup Script for Automated Batch Processing
Run this to verify your setup before pushing to GitHub
"""

import os
from batch_manager import (
    get_current_batch_number,
    get_all_links,
    get_batch_info,
    print_batch_status,
    prepare_current_batch,
    BATCH_SIZE,
    ALL_LINKS_FILE,
    CURRENT_LINKS_FILE,
    BATCH_PROGRESS_FILE,
    COUNTER_FILE
)

def check_file_exists(filename, required=True):
    """Check if a file exists and print status"""
    exists = os.path.exists(filename)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {filename}: {'Found' if exists else 'Missing'}")
    return exists

def check_file_content(filename, expected_type="number"):
    """Check if file has valid content"""
    if not os.path.exists(filename):
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if expected_type == "number":
            value = int(content)
            print(f"  → Value: {value}")
            return True
        elif expected_type == "links":
            lines = [line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
            print(f"  → Contains {len(lines)} links")
            return len(lines) > 0
    except Exception as e:
        print(f"  → Error reading: {e}")
        return False

def verify_setup():
    """Verify the complete setup"""
    print("\n" + "="*60)
    print("SETUP VERIFICATION")
    print("="*60 + "\n")
    
    all_good = True
    
    # Check all required files
    print("📁 Checking Required Files:")
    print("-" * 60)
    
    if check_file_exists(ALL_LINKS_FILE, required=True):
        if not check_file_content(ALL_LINKS_FILE, "links"):
            print("  ⚠️  File exists but appears empty or invalid!")
            all_good = False
    else:
        all_good = False
    
    if check_file_exists(BATCH_PROGRESS_FILE, required=True):
        check_file_content(BATCH_PROGRESS_FILE, "number")
    else:
        all_good = False
    
    if check_file_exists(COUNTER_FILE, required=True):
        check_file_content(COUNTER_FILE, "number")
    else:
        all_good = False
    
    check_file_exists(CURRENT_LINKS_FILE, required=False)
    
    # Check directories
    print("\n📂 Checking Directories:")
    print("-" * 60)
    check_file_exists("VIDEOS", required=True)
    check_file_exists("temp", required=False)
    check_file_exists(".github/workflows", required=True)
    
    # Check workflow file
    print("\n⚙️  Checking Workflow:")
    print("-" * 60)
    workflow_exists = check_file_exists(".github/workflows/downloader.yml", required=True)
    if not workflow_exists:
        all_good = False
    
    # Display batch information
    print("\n📊 Batch Information:")
    print("-" * 60)
    try:
        info = get_batch_info()
        print(f"Current Batch: {info['current_batch']} of {info['total_batches']}")
        print(f"Total Links: {info['total_links']}")
        print(f"Batch Size: {info['batch_size']}")
        print(f"Links Processed: {info['links_processed']}")
        print(f"Links Remaining: {info['links_remaining']}")
        print(f"Progress: {(info['links_processed'] / info['total_links'] * 100):.1f}%")
    except Exception as e:
        print(f"❌ Error getting batch info: {e}")
        all_good = False
    
    # Final status
    print("\n" + "="*60)
    if all_good:
        print("✅ SETUP VERIFICATION PASSED")
        print("="*60)
        print("\n🚀 You're ready to go!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Setup automated batch processing'")
        print("3. git push origin main")
        print("4. Enable GitHub Actions in your repository")
        print("5. Wait for automatic processing to begin!")
    else:
        print("❌ SETUP VERIFICATION FAILED")
        print("="*60)
        print("\n⚠️  Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("- Ensure all_links.txt contains your Instagram links")
        print("- Verify batch_progress.txt contains a number (e.g., 1)")
        print("- Verify counter.txt contains a number (e.g., 1)")
        print("- Check that .github/workflows/downloader.yml exists")
    print("\n")
    
    return all_good

def test_batch_preparation():
    """Test preparing the current batch"""
    print("\n" + "="*60)
    print("TESTING BATCH PREPARATION")
    print("="*60 + "\n")
    
    try:
        success = prepare_current_batch()
        if success:
            print("✅ Successfully prepared current batch!")
            print(f"📄 Check {CURRENT_LINKS_FILE} to see the current batch links")
            
            # Show first few links
            with open(CURRENT_LINKS_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"\n📋 First 5 links in current batch:")
            for i, line in enumerate(lines[:5], 1):
                print(f"  {i}. {line.strip()}")
            
            if len(lines) > 5:
                print(f"  ... and {len(lines) - 5} more")
        else:
            print("⚠️  No more batches to process or an error occurred")
    except Exception as e:
        print(f"❌ Error preparing batch: {e}")

def show_help():
    """Show help information"""
    print("\n" + "="*60)
    print("AUTOMATED BATCH PROCESSING - TEST SCRIPT")
    print("="*60 + "\n")
    
    print("This script helps you verify your setup before pushing to GitHub.")
    print("\nAvailable options:\n")
    print("  1. Verify Setup       - Check all files and configuration")
    print("  2. Show Status        - Display current batch status")
    print("  3. Test Batch Prep    - Test preparing current batch")
    print("  4. Show Help          - Display this help message")
    print("  5. Exit\n")

def main():
    """Main interactive menu"""
    while True:
        show_help()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                verify_setup()
            elif choice == "2":
                print_batch_status()
            elif choice == "3":
                test_batch_preparation()
            elif choice == "4":
                continue
            elif choice == "5":
                print("\n👋 Goodbye! Good luck with your automated downloads!\n")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1-5.\n")
            
            input("\nPress Enter to continue...")
            print("\n" * 2)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            input("Press Enter to continue...")

if __name__ == "__main__":
    # Quick check mode if run without interaction
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--verify":
            verify_setup()
        elif sys.argv[1] == "--status":
            print_batch_status()
        elif sys.argv[1] == "--test":
            test_batch_preparation()
        else:
            print("Usage: python test_setup.py [--verify|--status|--test]")
    else:
        main()
