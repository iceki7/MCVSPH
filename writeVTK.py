import numpy as np  
import vtk  
  

#GPT  
def writeVTK(vertices,velocities,fname,vecnum=3):

    #速度分量个数vecnum

    print(vertices.shape)#N 3
    print(velocities.shape)#N 3

    # 转换为VTK可以理解的格式
    vtk_points = vtk.vtkPoints()  
    for point in vertices:  
        vtk_points.InsertNextPoint(point)  
    
    # 假设网格由立方体单元组成（每个立方体由8个顶点和6个四边形面组成，但这里我们简化为顶点数据）  
    # 在实际中，你可能需要定义单元连接性（如使用vtkHexahedron）  
    # 但为了简单起见，我们仅保存顶点数据  
    
    # 创建UnstructuredGrid  
    grid = vtk.vtkUnstructuredGrid()  
    grid.SetPoints(vtk_points)  
    
    # 如果需要，可以添加单元（这里省略）  
    # ...  
    
    # 添加速度数组作为网格的属性  
    velocity_array = vtk.vtkFloatArray()  
    velocity_array.SetName("Velocity")  
    velocity_array.SetNumberOfComponents(vecnum)  

    velocity_array.SetNumberOfTuples(velocities.shape[0])  
    
    for i, velocity in enumerate(velocities):  
        if(vecnum==1):
        
            velocity_array.SetTuple1(i,*velocity)
        elif(vecnum==3):
            velocity_array.SetTuple3(i, *velocity)  
        else:
            assert(False)
      
    
    grid.GetPointData().AddArray(velocity_array)  
    
    # 写入VTK文件  
    writer = vtk.vtkXMLUnstructuredGridWriter()  
    writer.SetFileName(fname+".vtu")  
    writer.SetInputData(grid)  
    writer.Write()  
    
    print("VTK file written successfully.")