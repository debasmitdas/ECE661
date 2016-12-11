function imgs = loadImagesAdaboost(filePath, r, c)

% get the images in 'filePath'
files = dir([filePath '*.png']);
imgs=zeros(r,c,length(files));

for i=1:length(files)
    img=imread([filePath files(i).name]);
    imgs(:,:,i)=double(rgb2gray(img));
end
end