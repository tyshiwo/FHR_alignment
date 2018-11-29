require 'paths'
require 'image'
require 'math'
require 'io'
paths.dofile('util_128.lua')
paths.dofile('img_128.lua')
local matio = require 'matio'
--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1] == 'demo' or arg[1] == 'predict-test' then
    -- Test set annotations do not have ground truth part locations, but provide
    -- information about the location and scale of people in each image.
    -- a = loadAnnotations('test')
    --print('images/' .. a['images'][879])
    --debug.debug()
elseif arg[1] == 'predict-valid' or arg[1] == 'eval' then
    -- Validation set annotations on the other hand, provide part locations,
    -- visibility information, normalization factors for final evaluation, etc.
    a = loadAnnotations('valid')

else
    print("Please use one of the following input arguments:")
    print("    demo - Generate and display results on a few demo images")
    print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
    print("    predict-test - Generate predictions on the test set")
    print("    eval - Run basic evaluation on predictions from the validation set")
    return
end

name_s = 'Scenario1'
file_total = io.open('../data/' .. name_s .. '/fileList_300VW_' .. name_s .. '.txt', 'r')
-- 300VW pretrained model
m = torch.load('../models/model_300VW.t7')
output_size = 128
nStack = 4
pts_num = 68
dataset_name_ = '300VW'

for name__ in file_total:lines() do 
    print(name__)

    local matio = require 'matio'
    file = io.open('../data/' .. name_s .. '/fileList_300VW_' .. name__ .. '.txt', 'r')
    count__ = 0
    for line in file:lines() do 
        count__ = count__+1
    end

    file_num = count__
    preds_ = torch.Tensor(file_num,pts_num,2)
    --------------------------------------------------------------------------------
    -- Main loop
    --------------------------------------------------------------------------------
    RMSE=0
    counti = 1
    file = io.open('../data/' .. name_s .. '/fileList_300VW_' .. name__ .. '.txt', 'r')
    for line in file:lines() do 
        print(counti)
        path_ = unpack(line:split(" "))
        inp = image.load(path_)

    	y1=1
	    y2=256
    	x1=1
	    x2=256
    	inp_1=inp[{{},{y1,y2},{x1,x2}}]
        inp_image =image.scale(inp_1,256,256)
    	local out = m:forward(inp_image:view(1,3,256,256):cuda())
	    -- Get network output
        output_2=out[nStack][1]:float()
        local hm2 = torch.FloatTensor(pts_num,output_size,output_size)
        local hm = torch.FloatTensor(pts_num,output_size,output_size)
        --output_1=torch.CudaTensor(1,68,128,128)
        for kkk=1,pts_num do
            hm[{{kkk},{},{}}]:copy(output_2[{{kkk},{},{}}])
        end
        output = applyFn(function (x) return x:clone() end, hm)

        local flippedOut = m:forward(flip(inp_image):cuda())
        flippedOut_output_2=flippedOut[nStack][1]:float()
        flippedOut_output_1=torch.CudaTensor(pts_num,output_size,output_size)
        for kkk=1,pts_num do
            flippedOut_output_1[{{kkk},{},{}}]:copy(flippedOut_output_2[{{kkk},{},{}}])
        end
        flippedOut = applyFn(function (x) return flip(shuffleLR_68(x)) end, flippedOut_output_1)

        -- average 
        hm = applyFn(function (x,y) return x:add(y):div(2) end, output:float(), flippedOut:float())--:add(y):div(2)

        cutorch.synchronize()
        hm[hm:lt(0)] = 0

        local preds, preds_hm_ = postprocess_float_256(hm:view(1, hm:size(1), hm:size(2), hm:size(3)) )
    
        preds_[counti]:copy(preds*(256/output_size))
    	counti = counti+1


        collectgarbage()
    end

    matio.save('./results/' .. name_s .. '/preds_300VWtest_float_' .. pts_num .. 'pts_' .. name__ .. '_' .. dataset_name_ .. '_' .. output_size .. '.mat',preds_)

end
