"""
Test suite for the Dartboard Detector

This script tests the dartboard detector on multiple test images.
"""

import cv2
import sys
import os
from app.services.dartboard_detector import DartboardDetector


def test_image(detector, image_path, test_name):
    """
    Test the dartboard detector on a single image.
    
    Args:
        detector: DartboardDetector instance
        image_path: Path to test image
        test_name: Name of the test for display purposes
    """
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return False
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect dartboard
    print(f"\n🔍 Detecting dartboard...")
    dartboard_info = detector.detect_dartboard(image)
    
    if not dartboard_info:
        print(f"❌ No dartboard detected!")
        return False
    
    print(f"✓ Dartboard detected!")
    print(f"  Centre: {dartboard_info['centre']}")
    print(f"  Radius: {dartboard_info['radius']}px")
    print(f"  Method: {dartboard_info['method']}")
    
    # Detect segments
    print(f"\n🎯 Detecting segments...")
    segment_info = detector.detect_segments(image, dartboard_info)
    segment_info.update(dartboard_info)  # Merge info
    
    print(f"✓ Segments detected!")
    print(f"  Number of segments: {segment_info['num_segments']}")
    
    # Create visualisation
    print(f"\n🎨 Creating visualisation...")
    output_path = image_path.rsplit('.', 1)[0] + '_segments.jpg'
    result = detector.visualise_segments(image, segment_info, output_path)
    
    print(f"✓ Visualisation saved to: {output_path}")
    
    # Test scoring with example points
    print(f"\n🎯 Testing dart scoring:")
    test_points = [
        (dartboard_info['centre'][0], dartboard_info['centre'][1] - dartboard_info['radius'] // 2),  # Top area
        (dartboard_info['centre'][0] + dartboard_info['radius'] // 2, dartboard_info['centre'][1]),  # Right area
    ]
    
    for i, point in enumerate(test_points, 1):
        score_info = detector.get_segment_score(point, segment_info)
        print(f"\n  Test point {i} at {point}:")
        print(f"    Score: {score_info['score']}")
        print(f"    Region: {score_info['region']}")
        print(f"    Multiplier: {score_info['multiplier']}x")
        print(f"    Total: {score_info['total']} points")
    
    return True


def main():
    """Run all tests."""
    print(f"🎯 Dartboard Detector Test Suite")
    print(f"{'='*70}")
    
    # Create detector
    detector = DartboardDetector(debug=True)
    
    # Define test images
    test_images = [
        ("app/images/test1.jpg", "Test 1 - First Dartboard"),
        ("app/images/test2.jpg", "Test 2 - Second Dartboard"),
    ]
    
    # Run tests
    results = []
    for image_path, test_name in test_images:
        if os.path.exists(image_path):
            success = test_image(detector, image_path, test_name)
            results.append((test_name, success))
        else:
            print(f"\n⚠️ Warning: Test image not found: {image_path}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Test Summary")
    print(f"{'='*70}")
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

