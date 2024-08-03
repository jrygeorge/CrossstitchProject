# Cross-Stitch Pattern Generator

# NOTE :
# Make sure to delete EVERYTHING when youre done!! EKS (the control plane) isn't free !! (including the network interfaces)

[Flask App that generates a crossstitch pattern given an image and other parameters.
Hosted on EKS (in ECS) with S3 to store the files]

Takes an input image from the user, along with other parameters like output image dimensions and number of colours required,
and produces a cross-stitch pattern from it. Image is then upload to S3 and a pre-signed url is generated from which the user can download the file.

The whole thing is made into a docker container and then hosted with EKS

To change in the current code :
- Use to Distance Method from that one package
- Figure out how to create the pdf fully in memory
- Clean up the ugly front end

To add :
- OAuth to login a download saved files
- add a DB for the login stuff plus the files names (in S3)

-----

(yes i know it says crosstitch not crossstitch)
