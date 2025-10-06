"""
Test suite for the Dart Detector

This script tests the dart detector on dartboard images with darts.
"""

import cv2
import sys
import os
from app.services.dartboard_detector import DartboardDetector
from app.services.dart_detector import DartDetector


def test_dart_detection(dartboard_image_path, test_name):
    """
    Test the dart detector on a dartboard image.
    
    Args:
        dartboard_image_path: Path to dartboard image with darts
        test_name: Name of the test for display purposes
    """
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    
    # Load image
    print(f"📷 Loading image: {dartboard_image_path}")
    image = cv2.imread(dartboard_image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {dartboard_image_path}")
        return False
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # First, detect the dartboard
    print(f"\n🎯 Detecting dartboard...")
    dartboard_detector = DartboardDetector(debug=True)
    dartboard_info = dartboard_detector.detect_dartboard(image)
    
    if not dartboard_info:
        print(f"⚠️ Warning: No dartboard detected. Will search entire image for darts.")
        dartboard_info = None
    else:
        print(f"✓ Dartboard detected!")
        print(f"  Centre: {dartboard_info['centre']}")
        print(f"  Radius: {dartboard_info['radius']}px")
    
    # Detect darts
    print(f"\n🎯 Detecting darts...")
    dart_detector = DartDetector(debug=True)
    darts = dart_detector.detect_darts(image, dartboard_info)
    
    print(f"\n✓ Dart detection complete!")
    print(f"  Darts found: {len(darts)}")
    
    if darts:
        print(f"\n  Dart details:")
        for i, dart in enumerate(darts, 1):
            print(f"    Dart #{i}:")
            print(f"      Position: {dart['position']}")
            print(f"      Angle: {dart['angle']:.1f}°")
            print(f"      Confidence: {dart['confidence']:.2f}")
            print(f"      Length: {dart['length']}px")
            print(f"      Method: {dart['method']}")
        
        # If we have dartboard info, calculate scores
        if dartboard_info:
            print(f"\n🎯 Calculating dart scores:")
            segment_info = dartboard_detector.detect_segments(image, dartboard_info)
            segment_info.update(dartboard_info)
            
            total_score = 0
            for i, dart in enumerate(darts, 1):
                score_info = dartboard_detector.get_segment_score(dart['position'], segment_info)
                print(f"    Dart #{i}: {score_info['total']} points ({score_info['score']} × {score_info['multiplier']}, {score_info['region']})")
                total_score += score_info['total']
            
            print(f"\n  Total Score: {total_score} points")
    
    # Create visualisation
    print(f"\n🎨 Creating visualisation...")
    output_path = dartboard_image_path.rsplit('.', 1)[0] + '_darts_detected.jpg'
    result = dart_detector.visualise_darts(image, darts, output_path)
    
    print(f"✓ Visualisation saved to: {output_path}")
    
    return True


def main():
    """Run all dart detection tests."""
    print(f"🎯 Dart Detector Test Suite")
    print(f"{'='*70}")
    
    # Define test images
    # You can add your own test images here
    test_images = [
        ("app/images/test1.jpg", "Test 1 - Dartboard with Darts"),
        ("app/images/test2.jpg", "Test 2 - Dartboard with Darts"),
    ]
    
    # Run tests
    results = []
    for image_path, test_name in test_images:
        if os.path.exists(image_path):
            success = test_dart_detection(image_path, test_name)
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

