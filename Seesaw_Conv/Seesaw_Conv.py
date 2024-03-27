import torch
import torch.nn as nn
import time

class Seesaw_Conv(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            kernel_size: int = 3,
            device = 'cuda'
    ):
        super().__init__()

        if kernel_size % 2 == 0:
             raise ValueError(f'Recommended convolution kernel size should be odd numbers,  but got {kernel_size} instead')
        
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_elements = kernel_size ** 2
        self.padding = kernel_size // 2
        self.device = device

        self.find_ref = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=(self.kernel_elements-1)*in_channels,  # 9 // 2 * 2 = 9-1
                                  kernel_size=kernel_size, # 3
                                  padding=self.padding,
                                  groups=1
                                  )
        
        self.weighted_sum = nn.Conv2d(in_channels=self.kernel_elements * in_channels,
                              out_channels=in_channels,
                              kernel_size=1, 
                              groups=in_channels
                              )
        
        self.to(device)

    def find_symmetric_points(self,input):
        n = self.kernel_size
        center = n // 2
        center_tensor = torch.full_like(input, center)
        symmetric_tensor = 2 * center_tensor - input
        return symmetric_tensor
    
    def get_interpolated_feature(self,offset_coordinates,feature_map,mode='bilinear'):
        '''
        TODO: Implement this by CUDA
        '''
        # feature_map = feature_map.to(torch.float16)
        # offset_coordinates = offset_coordinates.to(torch.float16)

        bs,h,w,c,kernel_element,_ = offset_coordinates.shape
        interpolated_features = []
        for i in range(c):
            for j in range(kernel_element):
                grid = offset_coordinates[:,:,:,i,j,:]
                interpolated_feature = nn.functional.grid_sample(input=feature_map[:,i:i+1,:,:], grid=grid, mode=mode)
                interpolated_features.append(interpolated_feature) # interpolated_feature.shape = (bs,1,h,w)
        output = torch.stack(interpolated_features).view(bs,h,w,c,kernel_element)
        del interpolated_features
        return output
    
    def forward(self,x):
        bs,c,h,w = x.shape

        # Get offsets
        offsets = self.find_ref(x)
        offsets = offsets.permute(0,2,3,1) # b,h,w,c
        offsets = offsets.view(bs,h,w,c,-1,2) # Devide into c groups for inserting zeros in the middle
        symmetric_offsets = self.find_symmetric_points(offsets)
        center_points = torch.zeros(bs,h,w,c,1,2).to(self.device)
        offsets = torch.cat([offsets,center_points,symmetric_offsets], dim=-2)
        #print(offsets.shape) # bs,h,w,in_channel,kernel_element,2
        kernel_element = offsets.shape[4]

        # Offsets + coordinates
        h_grid,w_grid = torch.meshgrid(torch.linspace(0,1,h),torch.linspace(0,1,w),indexing='ij')
        grid = torch.stack((h_grid.unsqueeze(0).repeat(bs,1,1),
                            w_grid.unsqueeze(0).repeat(bs,1,1)),dim=-1
                            ).unsqueeze(3).unsqueeze(4).repeat(1,1,1,c,kernel_element,1).to('cuda')
        offsets_coordinates = grid.add(offsets)

        # TODO: Get feature by bilinear interpolate
        deformed_feature = self.get_interpolated_feature(offsets_coordinates,x)
        deformed_feature = deformed_feature.view(bs,h,w,-1).permute(0,3,1,2)
        output = self.weighted_sum(deformed_feature) # Weighted-sum along the channel dim

        return output.add(x)

class Seesaw_Conv_DCN(nn.Module):
    def __init__(
            self,
            kernel_size,
            in_channels,
            device='cuda'
            ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f'Recommended convolution kernel size should be odd numbers, but got {kernel_size} instead')
        
        self.kernel_size = kernel_size
        self.in_channel = in_channels
        self.n = kernel_size ** 2
        self.to(device)

        self.offset_conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size, out_channels=self.n // 2 * 2,padding=1)
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=in_channels)
    
    def find_symmetric_points(self,input):
        n = self.kernel_size
        center = n // 2
        center_tensor = torch.full_like(input, center)
        symmetric_tensor = 2 * center_tensor - input
        return symmetric_tensor
    
    def forward(self,x):
        bs,c,h,w = x.shape
        offset = self.offset_conv(x)
        offset = offset.view(bs,h,w,-1,2)
        symmetric_offset = self.find_symmetric_points(offset)
        center_points = torch.zeros(bs,h,w,1,2).to('cuda')
        offset = torch.cat([offset,center_points,symmetric_offset],dim=-2)
        print(offset[0,0,0,:,:])
        

if __name__ == '__main__':
    from torchsummary import summary
    import torchvision

    in_channels = 24
    x = torch.randn(8,in_channels,300,820).to('cuda')
    model = Seesaw_Conv(in_channels=in_channels,kernel_size=3).to('cuda')
    y = model(x)
    print(y.shape)
    # resnet18 = torchvision.models.resnet18(pretrained=True).to('cuda')
    # summary(model,(in_channels,300,820),batch_size=1)