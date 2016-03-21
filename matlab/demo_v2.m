% --------------------------------------------------------
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% Alter by Dean H
% 2016.3.1
% --------------------------------------------------------

function demo_v2(im_id_)
% Fast R-CNN demo (in matlab).

[folder, name, ext] = fileparts(mfilename('fullpath'));

caffe_path = fullfile(folder, '..', 'caffe-fast-rcnn', 'matlab', 'caffe');
addpath(caffe_path, 'SelectiveSearchCodeIJCV');
%selective_search_path = fullfile('/home', 'u514', 'DTask', 'rcnn', 'SelectiveSearch', ...
%                                   'SelectiveSearchCodeIJCV');
%selective_search_dependencies_path = fullfile(selective_search_path, 'Dependencies');
%addpath(selective_search_path, selective_search_dependencies_path);


use_gpu = true;
% You can try other models here:
def = fullfile(folder, '..', 'models', 'VGG16', 'test_flickr.prototxt');
net = fullfile(folder, '..', 'output', 'default', 'vgg16_flickr_trainval', ...
               'vgg16_fast_rcnn_iter_100000.caffemodel');
model = fast_rcnn_load_net(def, net, use_gpu);

%car_ind = 7;
%sofa_ind = 18;
%tv_ind = 20;
stellaartois_ind = 13;
google_ind = 1;
carlsberg = 20;
pepsi = 22;

AllLogo = [1:32];
AllLogoName = {'google', 'apple', 'adidas', 'hp', ...
               'stellaartois', 'paulaner', 'guiness',...
               'singha', 'cocacola', 'dhl', 'texaco', ...
               'fosters', 'fedex', 'aldi', 'chimay', ...
               'shell', 'becks', 'tsingtao', 'ford', ...
               'carlsberg', 'bmw', 'pepsi', 'esso', ...
               'heineken', 'erdinger', 'corona', 'milka', ...
               'ferrari', 'nvidia', 'rittersport', 'ups', 'starbucks'};

%demo(model, '000004', [car_ind], {'car'});
%demo(model, '001551', [sofa_ind, tv_ind], {'sofa', 'tvmonitor'});
tic;
%demo(model, '60524367', [pepsi], {'pepsi'});
demo(model, im_id_, AllLogo, AllLogoName);
toc;
fprintf('\n');

% ------------------------------------------------------------------------
function demo(model, im_id, cls_inds, cls_names)
% ------------------------------------------------------------------------
[folder, name, ext] = fileparts(mfilename('fullpath'));

%im_file = fullfile(folder, '..', 'data', 'demo', [im_id '.jpg']);
%im_file = fullfile('/home/u514/DTask', [im_id '.jpg']);

data_root_path = '/home/u514/fast-rcnn/data/data_flickr/VOC2007/JPEGImages';
output_root_path = fullfile('/home', 'u514', 'DTask', 'rcnn', 'data', 'testFlickr');
%im_name = '2213129757';
im_path = fullfile(data_root_path, [im_id '.jpg']);

im = imread(im_path);
tic;
boxes = selective_search_boxes(im);
toc;

boxes = single(boxes);
dets = fast_rcnn_im_detect(model, im, boxes);

THRESH = 0.5;
for j = 1:length(cls_inds)
  cls_ind = cls_inds(j);
  cls_name = cls_names{j};
  f = figure(cls_ind);
  %figure('Visible', 'off');
  I = dets{cls_ind}(:, end) >= THRESH;
  showboxes(f, im, dets{cls_ind}(I, :));
  title(sprintf('%s detections with p(%s | box) >= %.3f', ...
                cls_name, cls_name, THRESH))
            
  output_path = fullfile(output_root_path, cls_name);          
  if(~exist(output_path, 'dir'))
      mkdir(output_path);
      %saveas(gcf, [j, '.jpg']);
  end
  
  print(f, '-djpeg', fullfile(output_path, [im_id '.jpg']));          
  %fprintf('\n> Press any key to continue');
  %pause;
end
