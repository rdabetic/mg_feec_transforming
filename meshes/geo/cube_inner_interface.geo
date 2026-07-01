// cube_inner_interface.geo
SetFactory("OpenCASCADE");

// Unit cube and centered inner cube with edge length 0.5
Box(1) = {0, 0, 0, 1, 1, 1};
Box(2) = {0.25, 0.25, 0.25, 0.5, 0.5, 0.5};

BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; }

Physical Volume("domain") = {Volume{:}};

out_x0 = Surface In BoundingBox{-0.01, -0.1, -0.1,  0.01,  1.1,  1.1};
out_x1 = Surface In BoundingBox{ 0.99, -0.1, -0.1,  1.01,  1.1,  1.1};
out_y0 = Surface In BoundingBox{-0.1, -0.01, -0.1,  1.1,  0.01,  1.1};
out_y1 = Surface In BoundingBox{-0.1,  0.99, -0.1,  1.1,  1.01,  1.1};
out_z0 = Surface In BoundingBox{-0.1, -0.1, -0.01,  1.1,  1.1,  0.01};
out_z1 = Surface In BoundingBox{-0.1, -0.1,  0.99,  1.1,  1.1,  1.01};

Physical Surface("boundary") = {out_x0[], out_x1[], out_y0[], out_y1[], out_z0[], out_z1[]};

Mesh.CharacteristicLengthMin = 1;