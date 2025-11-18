## Dataset Access

The dataset used in this paper is hosted externally.

It is publicly available on Zenodo at:
https://zenodo.org/records/17631257


<h2>Dataset Acquisition</h2>

The shearographic data were acquired using a shearography sensor from Tenta Vision GmbH, which was mounted on a UR5 industrial robot arm (Universal Robots) to enable automated measurements.  

The measurement procedure employed spatial phase-shifting shearography, with a shear value of 1 mm oriented at a 45° angle relative to the specimen surface.  
Illumination was applied at a 0° angle, i.e., perpendicular to the specimen surface.  

The dataset comprises shearographic images recorded both with and without phase fringe patterns, including samples from defect-free as well as defective specimens.  

<h2>Dataset Description</h2>

<p>
This dataset contains shearographic measurement data of simulated kissing bonds in a polymer–rubber adhesive joint. 
Defects were introduced by inserting defined discontinuities within the adhesive layer, resulting in fully detached 
bonding regions. Additionally, the dataset includes reference measurements of non-defective specimens, some of which 
exhibit only global deformation, producing the characteristic phase-fringe patterns.
</p>

<p>The dataset comprises three categories of samples:</p>

<ul>
  <li><strong>Faulty</strong> – Defective samples with simulated kissing-bond anomalies</li>
  <li><strong>Good_clean</strong> – Non-defective samples without adhesive discontinuities and without global deformation</li>
  <li><strong>Good_stripes</strong> – Non-defective samples without adhesive discontinuities but with global deformation (phase fringes)</li>
</ul>

<p>
Ground-truth annotations (Bounding Box format, CSV, and YOLO format) are provided <strong>only</strong> for defective regions 
in the <em>Faulty</em> samples. All labels were created manually by a domain expert. Background and all non-defective regions 
remain unlabeled, resulting in partially annotated data suited for supervised, weakly supervised, or unsupervised 
learning approaches.
</p>

<hr>

<h2>CSV Files</h2>

<h3>dataset.csv</h3>
<p>This file contains metadata for all images in the dataset. Fields are <strong>semicolon-separated</strong>.</p>

<ul>
  <li><strong>image</strong> – image index</li>
  <li><strong>region</strong> – Identifier of the measurement position (multiple temporal recordings per position in Faulty samples)</li>
  <li><strong>time</strong> – Temporal order of image acquisition within each region</li>
  <li><strong>type</strong> – Sample category: <code>faulty</code>, <code>good_clean</code>, <code>good_stripes</code></li>
  <li><strong>split</strong> – Dataset split used in the associated publications</li>
  <li><strong>subset_a</strong> – Binary indicator (0/1) for inclusion in subset A (unsupervised learning without <code>good_stripes</code>)</li>
  <li><strong>subset_b</strong> – Binary indicator (0/1) for inclusion in subset B (unsupervised learning including <code>good_stripes</code>)</li>
</ul>

<hr>

<h3>labels.csv</h3>
<p>This file contains all ground-truth annotations for defective samples. Fields are <strong>semicolon-separated</strong>.</p>

<ul>
  <li><strong>image</strong> – Image index (corresponds to dataset.csv)</li>
  <li><strong>region</strong> – Measurement position (corresponds to dataset.csv)</li>
  <li><strong>time</strong> – Temporal acquisition order (corresponds to dataset.csv)</li>
  <li><strong>class</strong> – Annotated class label (29 classes total: 25 defect types + 4 artifact classes)</li>
  <li><strong>xmin, ymin, xmax, ymax</strong> – Bounding-box coordinates of the annotated region</li>
  <li><strong>split</strong> – Dataset split used in the associated publications</li>
</ul>
