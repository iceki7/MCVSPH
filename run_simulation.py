import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system import ParticleSystem,prm_fluidmodel,prm_nosim,trans,prm_quickexport,prm_exportbin
from particle_system import prm_sparse_export,gap
import time

ti.init(arch=ti.gpu, device_memory_fraction=0.9,debug=False,random_seed=1234,kernel_profiler=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    parser.add_argument('--cconvsceneidx',
                        default='9999',#prm
                        help='scene name')


    args = parser.parse_args()
    scene_path = args.scene_file
    cconvsceneidx=args.cconvsceneidx

    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]
    
    #zxc add
    scene_name = scene_path.split("\\")[-1].split(".")[0]
    if(prm_fluidmodel):
       #prm_
       scene_name="csm_"+str(cconvsceneidx)
    #    scene_name="csm_vr_"+str(cconvsceneidx)
    #    scene_name="specific_mp_"+str(cconvsceneidx)
    #    scene_name="specific_monte_"+str(cconvsceneidx)
    #    scene_name="specific_df_"+str(cconvsceneidx)
    #    scene_name="csm_df_sms_"+str(cconvsceneidx)
    #    scene_name="csm_mp_"+str(cconvsceneidx)


    print('[scene name]\t')
    print(scene_name)

       
    record_time = config.get_cfg("record_endTime")
    solid_num = config.get_cfg("solid_number")
    fluid_num = config.get_cfg("fluid_number")
    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_frames = config.get_cfg("exportFrame")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))

    #prm_
    # output_interval = int(0.004 / config.get_cfg("timeStepSize"))

    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
    color_series_prefix = "{}_output/fluid_particle_color_{}.ply".format(scene_name, "{}")
    vorticity_series_prefix = "{}_output/vorticity_object_{}.ply".format(scene_name, "{}")
    velocity_series_prefix = "{}_output/velocity_object_{}.ply".format(scene_name, "{}")

    if(prm_fluidmodel):
        #prm_
        output_frames=False
        output_ply=True
        output_obj=False


    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_output", exist_ok=True)


    ps = ParticleSystem(config, GGUI=True,cconvsceneidx=cconvsceneidx)
    solver = ps.build_solver()
    solver.initialize()

   
    if(prm_fluidmodel):
        #prm_
        solver.save_velocity=False
        solver.save_vorticity=False
        solver.save_color=False


        
    if(prm_quickexport):
        window = ti.ui.Window('SPH', (100, 100), show_window = True, vsync=False)   
    else:
        window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    cameraPos = config.get_cfg("cameraPos")
    cameraUp = config.get_cfg("cameraUp")
    cameraLookAt = config.get_cfg("cameraLookAt")
    camera.position(cameraPos[0],cameraPos[1],cameraPos[2])
    pointLightPos = config.get_cfg("pointLightPos")
    camera.up(cameraUp[0], cameraUp[1], cameraUp[2])
    camera.lookat(cameraLookAt[0], cameraLookAt[1], cameraLookAt[2])
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

    # Draw the lines for domain
    x_max, y_max, z_max = config.get_cfg("domainEnd")
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    #zxc 这个就可以当作法向量
    box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
    box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
    box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
    box_anchors[3] = ti.Vector([x_max, y_max, 0.0])

    box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
    box_anchors[5] = ti.Vector([0.0, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))

    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

    cnt = 0
    cnt_ply = 0

    if(prm_nosim):
        for i in range(substeps):
            solver.step()

    while window.running:

        if(prm_nosim==0):
            for i in range(substeps):
                solver.step()

        ps.copy_to_vis_buffer(invisible_objects=invisible_objects)
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
        elif ps.dim == 3:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            scene.point_light((pointLightPos[0], pointLightPos[1], pointLightPos[2]), color=(1.0, 1.0, 1.0))
            scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.particle_color)
            scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
            canvas.scene(scene)
    
        if output_frames:
            if cnt % output_interval == 0:
                window.save_image(f"{scene_name}_output_img/{cnt:06}.png")

        #prm
        if cnt % output_interval == 0:

            #prm swi----------------------sparse
            # if(prm_sparse_export):
            #     if(
            #     (cnt_ply%gap==1 or\
            #     cnt_ply%gap==2 or\
            #     cnt_ply%gap==3)):
                
                
                    

            if output_ply:
                obj_id = 0
                if(prm_fluidmodel):
                    obj_id=0
                obj_data = ps.dump(obj_id=obj_id)
                np_pos = obj_data["position"]
                writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
                writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                if(prm_exportbin):
                    writer.export_frame      (cnt_ply, series_prefix.format(0))
                else:
                    writer.export_frame_ascii(cnt_ply, series_prefix.format(0))#zxc edit taichi python file

                if solver.save_color:
                    np_color = obj_data["color"]
                    writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
                    writer.add_vertex_pos(np_color[:, 0], np_color[:, 1], np_color[:, 2])
                    writer.export_frame_ascii(cnt_ply, color_series_prefix.format(0))
                
                if solver.save_vorticity:
                    np_vorticity = obj_data["vorticity"]
                    writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
                    writer.add_vertex_pos(np_vorticity[:, 0], np_vorticity[:, 1], np_vorticity[:, 2])
                    writer.export_frame_ascii(cnt_ply, vorticity_series_prefix.format(0))

                if solver.save_velocity:
                    np_velocity = obj_data["velocity"]
                    writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
                    writer.add_vertex_pos(np_velocity[:, 0], np_velocity[:, 1], np_velocity[:, 2])
                    writer.export_frame_ascii(cnt_ply, velocity_series_prefix.format(0))
                
                if solid_num \
                    and prm_fluidmodel==0:
                    for i in range(solid_num):
                        obj_id = i+1
                        obj_data = ps.dump(obj_id=obj_id)
                        np_pos = obj_data["position"]
                        writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
                        writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                        writer.export_frame_ascii(cnt_ply, series_prefix.format(i+1))

            if output_obj:
                for r_body_id in ps.object_id_rigid_body:
                    with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                        e = ps.object_collection[r_body_id]["mesh"].export(file_type='obj')
                        f.write(e)

            #prm swi----------------------sparse endif

            cnt_ply += 1

        if(prm_nosim==0):
            cnt += 1

        #prm_
        if(cnt==1000):
            exit(0)
        window.show()