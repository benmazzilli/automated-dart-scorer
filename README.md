# Automated Dart Scorer

A computer vision solution with REST API for automated dartboard detection, segment identification, and dart scoring using OpenCV and FastAPI.

## Project Overview

This project implements advanced computer vision techniques to:
- **REST API**: Upload images via HTTP and get JSON scores (NEW!)
- **Integrated scoring pipeline**: Pass an image, get a complete score with clean visualisation
- Detect dartboards in images using multiple detection methods
- Identify all 20 segments with accurate boundaries
- Detect scoring rings (doubles, triples, bullseye)
- Detect darts in dartboard images
- Calculate scores for dart positions on the board
- Automatically detect board rotation and alignment
- Generate clean black & white visualisations showing dartboard outline and dart positions

## Features

### 🚀 REST API (NEW!)
- **FastAPI-powered**: Modern, fast, async REST API
- **Upload images**: POST endpoint accepts image files
- **JSON responses**: Get structured score data
- **Optional visualisation**: Endpoint returns annotated images
- **CORS enabled**: Ready for web frontend integration

### 🎯 Integrated Scoring Pipeline
- **Simple API**: Pass an image, get a score and visualisation
- **Automatic detection**: Finds dartboard, darts, and calculates scores
- **Clean visualisation**: Black and white dartboard outline with dart positions as dots
- **Complete score breakdown**: Individual dart scores and total

### Multi-Method Dartboard Detection
1. **Colour-based detection** - Fast detection using orange surround colour (Winmau boards)
2. **Contour-based detection** - Robust to lighting variations, finds circular shapes
3. **Hough circle detection** - Fallback method for difficult conditions

### Precise Segment Detection
- Detects all 20 dartboard segments with standard scoring layout
- Automatically aligns with board rotation using wire detection
- Identifies all scoring rings:
  - Inner bullseye (50 points)
  - Outer bull (25 points)
  - Triple ring (3x multiplier)
  - Double ring (2x multiplier)
  - Single regions (1x multiplier)

### Accurate Centre Detection
- Bullseye colour detection (red/green)
- Multiple refinement techniques for sub-pixel accuracy
- Validates centre position against detected rings

### Dart Detection (Simplified & Robust)
- **Primary method**: Detects bright, saturated coloured dart flights
- **Dartboard colour exclusion**: Intelligently filters out dark red/green segments
- **Balanced filtering** with multiple validation steps:
  - Size validation (200-2000 pixels)
  - Aspect ratio check (1.4:1 to 5:1)
  - Solidity and compactness analysis
  - Balanced brightness (≥100) and saturation (≥120) thresholds
- **High confidence threshold** (≥0.65) to avoid false positives
- **ROI constraint** using dartboard location
- Designed to find actual darts, not wires/segments/numbers

## Quick Start

1. **Install dependencies:**
```bash
uv sync
```

2. **Start the API server:**
```bash
uv run uvicorn app.main:app --reload
```

3. **Open the demo page:**
Open `demo.html` in your browser and upload a dartboard image!

The API will be running at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

### Dependencies
- Python >= 3.13
- OpenCV (opencv-python >= 4.8.0)
- NumPy >= 1.24.0
- SciPy >= 1.16.2 (for advanced rotation detection)
- FastAPI >= 0.115.0 (REST API)
- Uvicorn >= 0.32.0 (ASGI server)
- Python-multipart >= 0.0.9 (file upload support)

## Usage

### Running the FastAPI Server (NEW!)

Start the REST API server:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

**Interactive API documentation** is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/
```

#### 2. Score an Image (JSON only)
```bash
curl -X POST "http://localhost:8000/score" \
  -F "file=@path/to/dartboard.jpg"
```

**Response:**
```json
{
  "success": true,
  "message": "Scored 3 darts",
  "total_score": 140,
  "darts": [
    {
      "dart_id": 1,
      "position": {"x": 850, "y": 440},
      "score": 20,
      "multiplier": 3,
      "total": 60,
      "region": "triple",
      "confidence": 0.87
    },
    {
      "dart_id": 2,
      "position": {"x": 920, "y": 520},
      "score": 20,
      "multiplier": 2,
      "total": 40,
      "region": "double",
      "confidence": 0.82
    },
    {
      "dart_id": 3,
      "position": {"x": 980, "y": 500},
      "score": 20,
      "multiplier": 2,
      "total": 40,
      "region": "double",
      "confidence": 0.78
    }
  ]
}
```

#### 3. Score an Image with Visualisation
```bash
curl -X POST "http://localhost:8000/score-with-image" \
  -F "file=@path/to/dartboard.jpg" \
  -o scored_result.jpg
```

Returns the annotated image with score data in headers.

### Using the API from Python

**Option 1: Use the test client script (easiest)**
```bash
uv run python test_api.py app/images/test1.jpg
```

**Option 2: Use requests library directly**
```python
import requests

# Score an image
with open('dartboard.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/score',
        files={'file': f}
    )

result = response.json()
print(f"Total Score: {result['total_score']}")
print(f"Darts: {len(result['darts'])}")

for dart in result['darts']:
    print(f"Dart {dart['dart_id']}: {dart['total']} points")

# Score with visualisation
with open('dartboard.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/score-with-image',
        files={'file': f}
    )

with open('result.jpg', 'wb') as f:
    f.write(response.content)

# Score data is in headers
print(f"Total Score: {response.headers['X-Total-Score']}")
```

### Running Tests

**Test the integrated scorer** (RECOMMENDED - shows complete pipeline):

```bash
uv run python app/test_scorer.py
```

This demonstrates the complete scoring pipeline:
1. Loads test images from `app/images/`
2. Detects dartboard and darts automatically
3. Calculates total and individual dart scores
4. Creates clean black & white visualisation with dartboard outline and dart dots
5. Saves annotated images as `*_scored.jpg`

**Test the dartboard detector** with the provided test images:

```bash
uv run python app/test_dartboard_detector.py
```

This will:
1. Load test images from `app/images/`
2. Detect the dartboard in each image
3. Identify all segments and scoring rings
4. Create visualisations with segment labels
5. Test scoring at sample dart positions
6. Save annotated images as `*_segments.jpg`

**Test the dart detector** on dartboard images with darts:

```bash
# Full test suite
uv run python app/test_dart_detector.py

# Quick test on test1.jpg (2 blue darts)
uv run python app/quick_test.py
```

This will:
1. Load test images from `app/images/`
2. Detect both the dartboard and any darts in the image
3. Mark each detected dart with position and confidence
4. Calculate scores for each dart if dartboard is detected
5. Save annotated images as `*_darts_detected.jpg`

**Expected Results**:
- `test1.jpg`: Should detect 3 blue darts (2-4 acceptable)
- `test2.jpg`: Should detect 3 blue darts (2-4 acceptable)
- High confidence (>0.65) for true darts
- Minimal false positives (goal: <10% false positive rate)
- Balanced: Not too strict (missing real darts) or too loose (many false positives)

### Using the DartScorer Class (Recommended - Complete Pipeline)

```python
from app.services.dart_scorer import DartScorer

# Initialise scorer
scorer = DartScorer(debug=True)

# Simple one-line scoring
result = scorer.process_and_save('dartboard_with_darts.jpg', 'output_scored.jpg')

# Access results
print(f"Total Score: {result['total_score']} points")
print(f"Darts Detected: {len(result['dart_scores'])}")

for dart_score in result['dart_scores']:
    print(f"Dart #{dart_score['dart_id']}: {dart_score['total']} points "
          f"({dart_score['score']} × {dart_score['multiplier']}, {dart_score['region']})")

# Or for more control over the process:
import cv2

image = cv2.imread('dartboard_with_darts.jpg')

# Score the image
result = scorer.score_image(image)

if result['success']:
    print(f"Total: {result['total_score']} points")
    
    # Create custom visualisation
    visualisation = scorer.visualise_score(image, result, 'custom_output.jpg')
```

### Using the DartboardDetector Class

```python
from app.services.dartboard_detector import DartboardDetector
import cv2

# Initialise detector
detector = DartboardDetector(debug=True)

# Load image
image = cv2.imread('path/to/dartboard.jpg')

# Detect dartboard
dartboard_info = detector.detect_dartboard(image)

if dartboard_info:
    print(f"Dartboard found at {dartboard_info['centre']}")
    print(f"Radius: {dartboard_info['radius']}px")
    
    # Detect segments
    segment_info = detector.detect_segments(image, dartboard_info)
    segment_info.update(dartboard_info)
    
    # Create visualisation
    result = detector.visualise_segments(image, segment_info, 'output.jpg')
    
    # Score a dart at specific coordinates
    dart_position = (500, 300)  # x, y coordinates
    score_info = detector.get_segment_score(dart_position, segment_info)
    print(f"Score: {score_info['total']} points")
    print(f"Region: {score_info['region']}")
    print(f"Base score: {score_info['score']} x {score_info['multiplier']}")
```

### Using the DartDetector Class

```python
from app.services.dartboard_detector import DartboardDetector
from app.services.dart_detector import DartDetector
import cv2

# Initialise detectors
dartboard_detector = DartboardDetector(debug=True)
dart_detector = DartDetector(debug=True)

# Load image
image = cv2.imread('path/to/dartboard_with_darts.jpg')

# Detect dartboard first (optional but improves accuracy)
dartboard_info = dartboard_detector.detect_dartboard(image)

# Detect darts
darts = dart_detector.detect_darts(image, dartboard_info)

print(f"Found {len(darts)} darts")
for i, dart in enumerate(darts, 1):
    print(f"Dart {i}:")
    print(f"  Position: {dart['position']}")
    print(f"  Angle: {dart['angle']:.1f}°")
    print(f"  Confidence: {dart['confidence']:.2f}")
    
    # Calculate score if dartboard detected
    if dartboard_info:
        segment_info = dartboard_detector.detect_segments(image, dartboard_info)
        segment_info.update(dartboard_info)
        score_info = dartboard_detector.get_segment_score(dart['position'], segment_info)
        print(f"  Score: {score_info['total']} points")

# Create visualisation
result = dart_detector.visualise_darts(image, darts, 'output_darts.jpg')
```

## How It Works

### 1. Dart Detection (Simplified Approach)
The dart detector uses a focused approach to find actual darts while avoiding false positives:

**Strategy**: Dart flights are brighter than dartboard segments, with balanced filtering

1. **Colour Detection (HSV)**
   - Searches for high saturation (≥120) + high brightness (≥100)
   - Balanced thresholds to catch various dart colours and lighting
   - Finds brightly coloured regions

2. **Smart Dartboard Colour Exclusion**
   - Excludes DARK red segments (hue 0-15°/165-180°, value <150, sat >150)
   - Excludes DARK green segments (hue 35-85°, value <130, sat >140)
   - Allows bright red/green/blue dart flights to pass through

3. **Size Filtering**
   - Dart flights: 200-2000 pixels (accommodates different distances/angles)
   - Aspect ratio: 1.4:1 to 5:1 (typical dart flight proportions)

4. **Shape Validation**
   - Solidity check (≥0.4) - flights are fairly solid, not wire-like
   - Compactness check (0.2-0.95) - reasonable shape, not circles
   - Per-object brightness ≥100 and saturation ≥120

5. **Confidence Scoring**
   - Multi-factor scoring (aspect, area, solidity, saturation, brightness)
   - Only accepts detections with confidence ≥0.65
   - Returns dart tip position and orientation

### 2. Dartboard Detection
The system tries multiple detection methods in order of speed and reliability:

1. **Colour Detection** (fastest)
   - Converts image to HSV colour space
   - Detects orange surround ring
   - Calculates centre using moments
   - Refines centre by detecting bullseye colours

2. **Contour Detection** (robust)
   - Applies edge detection with CLAHE enhancement
   - Finds circular contours
   - Scores by circularity and size

3. **Hough Circles** (fallback)
   - Uses Hough Circle Transform
   - Selects largest detected circle

### 3. Centre Refinement
- Searches for red/green bullseye colours in the centre region
- Uses Hough circles as fallback
- Achieves sub-pixel accuracy for precise segment boundaries

### 4. Rotation Detection
- Samples edge intensity along multiple radii
- Detects wire positions using peak detection
- Calculates rotation offset from expected positions
- Aligns segment boundaries to actual board orientation

### 5. Segment Mapping
- Divides board into 20 equal segments (18° each)
- Maps to standard dartboard score layout:
  ```
  20 at top, then clockwise: 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
  ```

### 6. Scoring Calculation
For any point (x, y):
1. Calculate distance from centre to determine ring
2. Calculate angle to determine segment
3. Apply multiplier based on ring:
   - Inner bull: 50 points
   - Outer bull: 25 points
   - Triple ring: base score × 3
   - Double ring: base score × 2
   - Single: base score × 1

## Project Structure

```
automated-dart-scorer/
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application (NEW)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── dartboard_detector.py    # Dartboard detection
│   │   ├── dart_detector.py         # Dart detection
│   │   └── dart_scorer.py           # Integrated scoring pipeline
│   ├── images/                       # Test images
│   │   ├── test1.jpg
│   │   ├── test1_segments.jpg        # Dartboard visualisation
│   │   ├── test1_darts_detected.jpg  # Dart detection output
│   │   ├── test1_scored.jpg          # Complete scoring output
│   │   ├── test2.jpg
│   │   ├── test2_segments.jpg        # Dartboard visualisation
│   │   ├── test2_darts_detected.jpg  # Dart detection output
│   │   └── test2_scored.jpg          # Complete scoring output
│   ├── test_dartboard_detector.py    # Dartboard test suite
│   ├── test_dart_detector.py         # Dart detector test suite
│   └── test_scorer.py                # Integrated scorer test suite
├── demo.html                         # Web demo page (NEW)
├── test_api.py                       # API test client (NEW)
├── pyproject.toml                    # Project dependencies
├── uv.lock                           # Locked dependencies
└── README.md                         # This file
```

## Configuration

The `DartboardDetector` class uses standard dartboard dimensions based on official specifications:

- **Playing diameter**: 451mm (outer edge of double ring)
- **Double ring**: 162mm from centre (inner edge)
- **Triple ring**: 99-107mm from centre
- **Outer bull**: 15.9mm radius
- **Inner bull**: 6.35mm radius

These are automatically scaled to the detected board size in the image.

## Example Output

### Integrated Scorer Output

When running the integrated scorer (`test_scorer.py`), you'll see:

```
🎯 Integrated Dart Scorer Test Suite
======================================================================
Testing: Test 1 - Dartboard with Darts
======================================================================
📷 Loading image: app/images/test1.jpg
✓ Image loaded: 1920x1080 pixels

🎯 Scoring image...
✓ Dartboard found at (960, 540) with radius 450px
✓ Found 3 darts
✓ Total score: 120 points

======================================================================
📊 SCORE RESULTS
======================================================================
  Total Score: 120 points
  Darts Detected: 3

  Individual Dart Scores:
    Dart #1:
      Position: (850, 440)
      Score: 60 points (20 × 3)
      Region: triple
      Confidence: 0.87
    Dart #2:
      Position: (920, 520)
      Score: 40 points (20 × 2)
      Region: double
      Confidence: 0.82
    Dart #3:
      Position: (980, 500)
      Score: 20 points (20 × 1)
      Region: single
      Confidence: 0.78

🎨 Creating clean black & white visualisation...
✓ Visualisation saved to: app/images/test1_scored.jpg
  - Clean white background
  - Black dartboard outline with segments
  - Small black dots for dart positions
  - Minimal text labels
```

### Dartboard Detector Output

When running dartboard detector tests, you'll see:

```
🎯 Dartboard Detector Test Suite
======================================================================
Testing: Test 1 - First Dartboard
======================================================================
📷 Loading image: app/images/test1.jpg
✓ Image loaded: 1920x1080 pixels

🔍 Detecting dartboard...
✓ Dartboard detected!
  Centre: (960, 540)
  Radius: 450px
  Method: colour

🎯 Detecting segments...
✓ Segments detected!
  Number of segments: 20

🎨 Creating visualisation...
Detected 20 wires, rotation offset: 2.3°
✓ Visualisation saved to: app/images/test1_segments.jpg

🎯 Testing dart scoring:
  Test point 1 at (960, 315):
    Score: 20
    Region: triple
    Multiplier: 3x
    Total: 60 points
```

## Current Status

This project is under active development on the `feature/computer-vision-scoring` branch.

### Completed
- ✅ Multi-method dartboard detection
- ✅ Precise centre detection with bullseye refinement
- ✅ Automatic rotation detection and alignment
- ✅ Segment boundary detection
- ✅ Scoring ring identification
- ✅ Point-to-score calculation
- ✅ Comprehensive test suite
- ✅ Visual debugging and annotation
- ✅ Dart tip detection from images
- ✅ Multiple dart detection methods
- ✅ Confidence-based dart filtering
- ✅ Integrated scoring pipeline
- ✅ Clean black & white dartboard visualisation with dart positions
- ✅ **REST API with FastAPI** (NEW!)
- ✅ **JSON response format** (NEW!)
- ✅ **Image upload endpoints** (NEW!)

### Future Enhancements
- 🔄 Video stream processing
- 🔄 Real-time scoring interface
- 🔄 Game mode implementation (501, Cricket, etc.)
- 🔄 Multi-player support
- 🔄 Statistics tracking
- 🔄 Web frontend
- 🔄 Mobile app support

## Troubleshooting

**No dartboard detected:**
- Ensure the dartboard is clearly visible and well-lit
- Try different detection methods by checking debug output
- The dartboard should be roughly circular in the image

**Incorrect segment boundaries:**
- Check that the centre detection is accurate (visualise output)
- Ensure wire detection is working (check debug logs)
- May need to adjust rotation detection parameters for non-standard boards

**Wrong scores:**
- Verify centre position is accurate
- Check that the radius detection is correct
- Ensure the board is not significantly distorted in the image

## Contributing

This is an active development project. Contributions and suggestions are welcome!

## License

This project is for educational and personal use.
