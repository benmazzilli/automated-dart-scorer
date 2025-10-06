"""
Test Suite for Integrated Dart Scorer

Demonstrates the complete scoring pipeline:
1. Pass an image
2. Detect dartboard and darts
3. Calculate scores
4. Generate visualisation with dartboard segments and dart positions as dots
"""

import cv2
import sys
import os
from app.services.dart_scorer import DartScorer


def test_image_scoring(image_path: str, test_name: str):
    """
    Test the integrated scoring system on a dartboard image.
    
    Args:
        image_path: Path to dartboard image with darts
        test_name: Name of the test for display purposes
    """
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    
    # Initialise scorer
    scorer = DartScorer(debug=True)
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return False
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Score the image
    print(f"\n🎯 Scoring image...")
    result = scorer.score_image(image)
    
    if not result['success']:
        print(f"❌ Scoring failed: {result['message']}")
        return False
    
    # Display results
    print(f"\n✓ Scoring complete!")
    print(f"\n{'='*70}")
    print(f"📊 SCORE RESULTS")
    print(f"{'='*70}")
    print(f"  Total Score: {result['total_score']} points")
    print(f"  Darts Detected: {len(result['dart_scores'])}")
    
    if result['dart_scores']:
        print(f"\n  Individual Dart Scores:")
        for dart_score in result['dart_scores']:
            print(f"    Dart #{dart_score['dart_id']}:")
            print(f"      Position: {dart_score['position']}")
            print(f"      Score: {dart_score['total']} points "
                  f"({dart_score['score']} × {dart_score['multiplier']})")
            print(f"      Region: {dart_score['region']}")
            print(f"      Confidence: {dart_score['confidence']:.2f}")
    
    # Create visualisation
    print(f"\n🎨 Creating clean black & white visualisation...")
    output_path = image_path.rsplit('.', 1)[0] + '_scored.jpg'
    scorer.visualise_score(image, result, output_path)
    
    print(f"✓ Visualisation saved to: {output_path}")
    print(f"  - Clean white background")
    print(f"  - Black dartboard outline with segments")
    print(f"  - Small black dots for dart positions")
    print(f"  - Minimal text labels")
    
    return True


def demo_api_usage():
    """Demonstrate the simple API usage."""
    print(f"\n{'='*70}")
    print(f"API USAGE EXAMPLE")
    print(f"{'='*70}")
    
    print("""
# Simple one-line scoring:
from app.services.dart_scorer import DartScorer

scorer = DartScorer()
result = scorer.process_and_save('dartboard.jpg', 'output.jpg')

print(f"Total Score: {result['total_score']} points")
print(f"Darts: {len(result['dart_scores'])}")

# Or for more control:
import cv2

image = cv2.imread('dartboard.jpg')
result = scorer.score_image(image)
visualisation = scorer.visualise_score(image, result, 'output.jpg')

print(f"Score: {result['total_score']}")
    """)


def main():
    """Run the integrated scoring tests."""
    print(f"🎯 Integrated Dart Scorer Test Suite")
    print(f"{'='*70}")
    print(f"This demonstrates the complete scoring pipeline:")
    print(f"  1. Image input")
    print(f"  2. Dartboard & dart detection")
    print(f"  3. Score calculation")
    print(f"  4. Visualisation with dartboard segments and dart positions")
    
    # Define test images
    test_images = [
        ("app/images/test1.jpg", "Test 1 - Dartboard with Darts"),
        ("app/images/test2.jpg", "Test 2 - Dartboard with Darts"),
    ]
    
    # Run tests
    results = []
    for image_path, test_name in test_images:
        if os.path.exists(image_path):
            success = test_image_scoring(image_path, test_name)
            results.append((test_name, success))
        else:
            print(f"\n⚠️ Warning: Test image not found: {image_path}")
            results.append((test_name, False))
    
    # Show API usage
    demo_api_usage()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        print("\nCheck the generated *_scored.jpg files to see:")
        print("  - Dartboard segments visualised")
        print("  - Dart positions marked as dots")
        print("  - Individual and total scores displayed")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

