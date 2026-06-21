
V = reshape(ebsd3.phaseId, size(ebsd3));

color = repcell(ones(1,3),numel(ebsd3.CSList),1)

color(ebsd3.indexedPhasesId) = cellfun(@(cs) cs.color,...
  ebsd3.CSList(ebsd3.indexedPhasesId),'UniformOutput',false)

color = vertcat(color{:})

figure(3)

%vol = volshow(V,Colormap=color);

vol = volshow(V,OverlayData=V);

vol.AlphaData = ebsd3.isIndexed

%%

v2 = 0.5*double(reshape(ebsd3.isIndexed,size(ebsd3)));

l2 = reshape(ebsd3.phaseId,size(ebsd3))-1;

h = labelvolshow(l2,v2,'BackgroundColor','w','LabelColor',ebsd3.colorList);



%%

v2 = double(reshape(ebsd3.isIndexed,size(ebsd3)));
l2 = uint8(reshape(ebsd3.phaseId,size(ebsd3)))-1;

viewer = viewer3d(BackgroundColor="black");
%vol = volshow(v2,OverlayData=l2, ...
%    OverlayColormap=ebsd3.colorList,Parent=viewer);

vol = volshow(v2,OverlayData=l2,Parent=viewer);
vol.AlphaData = ebsd3.isIndexed


%%

V = reshape(ebsd3.colorList(ebsd3.phaseId,:),[size(ebsd3),3]);
vol = volshow(V,RenderingStyle="SlicePlanes")
%vol.AlphaData = ebsd3.isIndexed

%% phase display
viewer = viewer3d

%try,delete(vol); end
%v2 = 255*uint8(ebsd3.isIndexed);
v2 = double(ebsd3.isIndexed);
%v2 = rand(size(ebsd3));
l2 = uint8(reshape(ebsd3.phaseId,size(ebsd3)))-1;
vol = volshow(v2,OverlayData = l2,...
  RenderingStyle="SlicePlanes",Parent=viewer);
  %OverlayDisplayRange = [1,255],...
vol.OverlayColormap(1:10,:)=ebsd3.colorList;

vol.OverlayAlpha=1;
%vol.OverlayAlphamap = ones(255,1); vol.OverlayAlphamap(1)=0;
%vol.AlphaData = double(ebsd3.isIndexed);
%vol.OverlayRenderingStyle = "LabelOverlay";

%% orientation display

color = reshape(ebsd3.colorList(ebsd3.phaseId,:),[size(ebsd3),3]);

vol = volshow(color,'RenderingStyle','SlicePlanes');

vol.AlphaData = double(ebsd3.isIndexed);

vol.AlphaData = double(reshape(ebsd3.phaseId,size(ebsd3)) >5)


%%

D = reshape(ebsd3.phaseId,size(ebsd3));


viewer = viewer3d;
vol = volshow(V,Parent=viewer, ...
    RenderingStyle="CinematicRendering", ...
    Colormap=colormap);



%%


load mri
V = squeeze(D);


%%

intensity = [0 20 40 120 220 1024];
alpha = [0 0 0.15 0.3 0.38 0.5];
color = [0 0 0; 43 0 0; 103 37 20; 199 155 97; 216 213 201; 255 255 255]/255;
queryPoints = linspace(min(intensity),max(intensity),256);
alphamap = interp1(intensity,alpha,queryPoints)';
colormap = interp1(intensity,color,queryPoints);

%%

sx = 1;
sy= 1;
sz = 2.5;
A = [sx 0 0 0; 0 sy 0 0; 0 0 sz 0; 0 0 0 1];

%%

tform = affinetform3d(A);

%%
vol = volshow(V,Colormap=colormap,Alphamap=alphamap,Transformation=tform);

%%

viewer = viewer3d;
vol = volshow(V,Parent=viewer, ...
    RenderingStyle="CinematicRendering", ...
    Colormap=colormap, ...
    Alphamap=alphamap, ...
    Transformation=tform);