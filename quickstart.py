import os
import sys

def check_environment():
    print("=" * 60)
    print("RAG Chatbot - Setup Checker")
    print("=" * 60)

    checks_passed = True

    print("\nChecking Python version...")
    if sys.version_info >= (3, 9):
        print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor} (OK)")
    else:
        print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} (Need 3.9+)")
        checks_passed = False

    print("\nChecking required packages...")
    required_packages = [
        'langchain', 'streamlit', 'chromadb',
        'pypdf', 'dotenv', 'openai'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package if package != 'dotenv' else 'dotenv')
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing_packages.append(package)
            checks_passed = False

    if missing_packages:
        print(f"\n   Install missing packages with:")
        print(f"   pip install -r requirements.txt")

    print("\nChecking environment configuration...")
    if os.path.exists('.env'):
        print("   ✓ .env file found")

        from dotenv import load_dotenv
        load_dotenv()

        openai_key = os.getenv('OPENAI_API_KEY')
        hf_key = os.getenv('HUGGINGFACE_API_KEY')

        if openai_key and openai_key != 'your_openai_api_key_here':
            print("   ✓ OPENAI_API_KEY configured")
        elif hf_key and hf_key != 'your_huggingface_api_key_here':
            print("   ✓ HUGGINGFACE_API_KEY configured")
        else:
            print("   ⚠ No valid API key found in .env")
            print("   → Copy .env.example to .env and add your API key")
            checks_passed = False
    else:
        print("   ✗ .env file not found")
        print("   → Copy .env.example to .env and add your API key")
        checks_passed = False

    print("\nChecking data directory...")
    if os.path.exists('data'):
        pdf_files = [f for f in os.listdir('data') if f.endswith('.pdf')]
        if pdf_files:
            print(f"   ✓ Found {len(pdf_files)} PDF file(s)")
        else:
            print("   ⚠ No PDF files in data/ directory")
            print("   → Add PDF files to data/ directory before ingestion")
    else:
        os.makedirs('data')
        print("   ✓ Created data/ directory")
        print("   → Add PDF files to data/ directory")

    print("\nChecking vector store...")
    if os.path.exists('vectorstore') and os.listdir('vectorstore'):
        print("   ✓ Vector store exists")
    else:
        print("   ⚠ Vector store not initialized")
        print("   → Run ingestion first: python ingest.py")

    print("\n" + "=" * 60)

    if checks_passed:
        print("✓ All checks passed! Ready to launch.")
        print("\nNext steps:")
        print("  1. Add PDF files to data/ directory")
        print("  2. Run: python ingest.py")
        print("  3. Run: streamlit run app.py")
    else:
        print("⚠ Some checks failed. Please fix the issues above.")

    print("=" * 60)

    return checks_passed


def main():
    print("\n")

    if check_environment():
        print("\nWould you like to start the Streamlit app now? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response == 'y':
                print("\nStarting Streamlit app...\n")
                os.system("streamlit run app.py")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    else:
        print("\nRun this script again after fixing the issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
