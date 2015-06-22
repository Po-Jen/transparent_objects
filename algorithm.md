#The explanation of transparent object detection algorithm

##Training stage

###Model construction
This algorithm is a template-based method. So it needs a predefined 3D model of the transparent object you want to detect. 

You can provide your model, either by KinectFusion to construct one or download from web.

###Silhouettes generation
Once you have the model, the training algorithm will turn the 3D model in different viewpoints, and store the silhouette of each viewpoint in database.(The silhouettes are different, since they are in different viewpoints) This is because in testing stage, the test silhouette (the silhouette in the scene) can be matched to the silhouettes in database(training silhouettes).

The best match provide hint about the pose of test silhouette. You can imagine that you don't know the real pose of test silhouette, so you then try to turn the 3D model around and see what's the pose that produce most alike silhouette.

##Testing stage
### Test silhouette detection
In testing stage, the depth map from Kinect is used to search the candidate region of transparency. Because transparent object produce NaN in depth map(black region in depth map), so the NaN region in depth map can be candidate of transparency. 

However, due to Kinect's characteristic, there are many small NaN region in depth map. So some morphological operations are used to rule out these regions. (Like a process of denoising)

Once the candidate region is fetched, this can be the foreground guess which can be fed to GrabCut. Grabcut then crop the test silhouette.

### Pose estimation(Estimate Rx, Ry, Rz and T)
The test silhouette is now acquired, it can be matched with training silhouettes in the database by applying Proscrutes Analysis. Similarity transforms of each training silhouette to test silhouette can be computed. (In this step, Rx&Ry are computed)

Next step is to find Rz and T by applying weak projection model. After this step, Rx, Ry, Rz and T of each training silhouette are all computed.

Finally, Chamfer matching is used to score each training silhouette. And a rough pose estimation can thus be computed.


The main idea is written, and details are neglected for the ease of understanding.
