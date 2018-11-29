paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

        --first HG
        local S1_pose_hg = hourglass(6,opt.nFeats,inter)
        -- Residual layers at output resolution
        local S1_pose_fea_1 = S1_pose_hg
        for j = 1,opt.nModules do S1_pose_fea_1 = Residual(opt.nFeats,opt.nFeats)(S1_pose_fea_1) end
        -- Linear layer to produce first set of predictions
        S1_pose_fea_2 = lin(opt.nFeats,opt.nFeats,S1_pose_fea_1) 
        -- Predicted heatmaps
        local S1_pose_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S1_pose_fea_2)    
        table.insert(out,S1_pose_out)
        local S1_occl_fea_1 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S1_pose_out)
        local S1_occl_fea_2 = nn.CAddTable()({S1_pose_fea_1,inter,S1_occl_fea_1})
        local S1_occl_hg = hourglass(6,opt.nFeats,S1_occl_fea_2)
        local S1_occl_fea_3 = S1_occl_hg
        for j = 1,opt.nModules do S1_occl_fea_3 = Residual(opt.nFeats,opt.nFeats)(S1_occl_fea_3) end
        S1_occl_fea_4 = lin(opt.nFeats,opt.nFeats,S1_occl_fea_3)
        local S1_occl_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S1_occl_fea_4)
        table.insert(out,S1_occl_out)

        --second HG
        -- Add predictions back 
        local S2_pose_fea_1 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(S1_pose_fea_1)
        local S2_pose_fea_2 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S1_pose_out)
        local S2_pose_fea_3 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S1_occl_out)
        inter2 = nn.CAddTable()({inter, S2_pose_fea_1, S2_pose_fea_2,S2_pose_fea_3})      
        local S2_pose_hg = hourglass(6,opt.nFeats,inter2)
        -- Residual layers at output resolution
        local S2_pose_fea_4 = S2_pose_hg
        for j = 1,opt.nModules do S2_pose_fea_4 = Residual(opt.nFeats,opt.nFeats)(S2_pose_fea_4) end
        -- Linear layer to produce first set of predictions
        S2_pose_fea_5 = lin(opt.nFeats,opt.nFeats,S2_pose_fea_4)
        -- Predicted heatmaps
        local S2_pose_fea_6 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(S2_pose_fea_5)
        local S2_pose_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S2_pose_fea_6)
        table.insert(out,S2_pose_out)
        local S2_occl_fea_1 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S2_pose_out)
        local S2_occl_fea_2 = nn.CAddTable()({S2_pose_fea_4,inter,S2_occl_fea_1})
        local S2_occl_fea_3 = hourglass(6,opt.nFeats,S2_occl_fea_2)
        local S2_occl_fea_4 = S2_occl_fea_3
        for j = 1,opt.nModules do S2_occl_fea_4 = Residual(opt.nFeats,opt.nFeats)(S2_occl_fea_4) end
        S2_occl_fea_5 = lin(opt.nFeats,opt.nFeats,S2_occl_fea_4)
        local S2_occl_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S2_occl_fea_5)
        table.insert(out,S2_occl_out)

        
        --third HG
        local S3_pose_fea_1 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(S2_pose_fea_4)
        local S3_pose_fea_2 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S2_pose_out)
        local S3_pose_fea_3 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S2_occl_out)
        inter3 = nn.CAddTable()({inter, S3_pose_fea_1, S3_pose_fea_2,S3_pose_fea_3})      
        local S3_pose_hg = hourglass(6,opt.nFeats,inter3)
        -- Residual layers at output resolution
        local S3_pose_fea_4 = S3_pose_hg
        for j = 1,opt.nModules do S3_pose_fea_4 = Residual(opt.nFeats,opt.nFeats)(S3_pose_fea_4) end
        -- Linear layer to produce first set of predictions
        S3_pose_fea_5 = lin(opt.nFeats,opt.nFeats,S3_pose_fea_4)
        -- Predicted heatmaps
        local S3_pose_fea_6 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(S3_pose_fea_5)
        local S3_pose_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S3_pose_fea_6)
        table.insert(out,S3_pose_out)
        local S3_occl_fea_1 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S3_pose_out)
        local S3_occl_fea_2 = nn.CAddTable()({S3_pose_fea_4,inter,S3_occl_fea_1})
        local S3_occl_fea_3 = hourglass(6,opt.nFeats,S3_occl_fea_2)
        local S3_occl_fea_4 = S3_occl_fea_3
        for j = 1,opt.nModules do S3_occl_fea_4 = Residual(opt.nFeats,opt.nFeats)(S3_occl_fea_4) end
        S3_occl_fea_5 = lin(opt.nFeats,opt.nFeats,S3_occl_fea_4)
        local S3_occl_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S3_occl_fea_5)
        table.insert(out,S3_occl_out)  
        
        
        --fouth HG
        local S4_pose_fea_1 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(S3_pose_fea_4)
        local S4_pose_fea_2 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S3_pose_out)
        local S4_pose_fea_3 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S3_occl_out)
        inter3 = nn.CAddTable()({inter, S4_pose_fea_1, S4_pose_fea_2,S4_pose_fea_3})      
        local S4_pose_hg = hourglass(6,opt.nFeats,inter3)
        -- Residual layers at output resolution
        local S4_pose_fea_4 = S4_pose_hg
        for j = 1,opt.nModules do S4_pose_fea_4 = Residual(opt.nFeats,opt.nFeats)(S4_pose_fea_4) end
        -- Linear layer to produce first set of predictions
        S4_pose_fea_5 = lin(opt.nFeats,opt.nFeats,S4_pose_fea_4)
        -- Predicted heatmaps
        local S4_pose_fea_6 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(S4_pose_fea_5)
        local S4_pose_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S4_pose_fea_6)
        table.insert(out,S4_pose_out)
        local S4_occl_fea_1 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(S4_pose_out)
        local S4_occl_fea_2 = nn.CAddTable()({S4_pose_fea_4,inter,S4_occl_fea_1})
        local S4_occl_fea_3 = hourglass(6,opt.nFeats,S4_occl_fea_2)
        local S4_occl_fea_4 = S4_occl_fea_3
        for j = 1,opt.nModules do S4_occl_fea_4 = Residual(opt.nFeats,opt.nFeats)(S4_occl_fea_4) end
        S4_occl_fea_5 = lin(opt.nFeats,opt.nFeats,S4_occl_fea_4)
        local S4_occl_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(S4_occl_fea_5)
        table.insert(out,S4_occl_out)  
        --local finaloutput = nn.CAddTable()({tmpOut2,occlusion_7})




    -- Final model
    local netG = nn.gModule({inp}, out)

    return netG

end
