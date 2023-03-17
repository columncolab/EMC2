import warnings
warnings.filterwarnings("ignore")

import netCDF4
import glob
import copy 
import pandas as pd
import scipy
from scipy import interpolate
import numpy as np
import xarray as xr

import act

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import matplotlib.cm as cm



def get_radar_lidar_signals(my_e3sm):

    atb_total_4D=np.empty_like(my_e3sm.ds.sub_col_beta_att_tot.values)
    atb_mol_4D=np.empty_like(my_e3sm.ds.sub_col_beta_att_tot.values)

    z_full_km_3D=np.empty_like(my_e3sm.ds.Z3.values)
    z_half_km_3D=np.empty_like(my_e3sm.ds.Z3.values)

    # newly added for Ze att tot
    Ze_att_total_4D=np.empty_like(my_e3sm.ds.sub_col_beta_att_tot.values)


    subcolum_num=len(my_e3sm.ds.subcolumn)
    time_num=len(my_e3sm.ds.time)
    col_num=len(my_e3sm.ds.ncol)
    lev_num=len(my_e3sm.ds.lev)

    for i in np.arange(subcolum_num):
        for j in np.arange(time_num):    
            for k in np.arange(col_num):  
                atb_total_4D[i,j,:,k],atb_mol_4D[i,j,:,k],\
                z_full_km_3D[j,:,k],z_half_km_3D[j,:,k],\
                Ze_att_total_4D[i,j,:,k]=get_ATB_ATBmol(my_e3sm,i,j,k)

    return atb_total_4D,atb_mol_4D,z_full_km_3D,z_half_km_3D,Ze_att_total_4D



def get_ATB_ATBmol(my_e3sm,subcolum_index,time_index,ncol_index):
    
    sub_col_beta_att_tot_sub=my_e3sm.ds.sub_col_beta_att_tot[subcolum_index,time_index,:,ncol_index]#.values

    sigma_180_vol_sub=my_e3sm.ds.sigma_180_vol[time_index,:,ncol_index]#.values

    tau_sub=my_e3sm.ds.tau[time_index,:,ncol_index]#.values

    sub_col_beta_p_tot_sub=my_e3sm.ds.sub_col_beta_p_tot[subcolum_index,time_index,:,ncol_index]#.values

    sub_col_OD_tot_sub=my_e3sm.ds.sub_col_OD_tot[subcolum_index,time_index,:,ncol_index]#.values
    
    sub_col_Ze_att_tot_sub=my_e3sm.ds.sub_col_Ze_att_tot[subcolum_index,time_index,:,ncol_index]#.values
     
    
    sub_col_beta_att_tot_sub_est=(sigma_180_vol_sub+sub_col_beta_p_tot_sub.fillna(0))*tau_sub*np.exp(-2.*1.*sub_col_OD_tot_sub.fillna(0))

    # get the right singal values
    sub_atb_mol=(sigma_180_vol_sub*tau_sub).values
    sub_atb_total=sub_col_beta_att_tot_sub.values
    
    sub_ze_att_total=sub_col_Ze_att_tot_sub.values
   
    z_filed_m=my_e3sm.ds.Z3.values[time_index,:,ncol_index]/1000. # km
    
    zfull=copy.deepcopy(z_filed_m)
    zhalf=np.zeros_like(zfull)
    zhalf[:-1]=zfull[:-1]+np.diff(zfull)/2.
    zhalf[-1]=zfull[-1]    
    
    
    return sub_atb_total,sub_atb_mol,z_filed_m,zhalf,sub_ze_att_total


def calculate_SR(atb_total_4D,atb_mol_4D,subcolum_num,time_num,lev_num,z_full_km_3D,z_half_km_3D,\
                 Ncolumns,Npoints,Nlevels,Nglevels,col_num,newgrid_bot,newgrid_top):
    


    R_UNDEF      = -1.0E30   #   Missing value
    b=copy.deepcopy(atb_total_4D)
    atb_total_4D_num=np.nan_to_num(b, nan=R_UNDEF)
    c=copy.deepcopy(atb_mol_4D)
    atb_mol_4D_num=np.nan_to_num(c, nan=R_UNDEF)

    Regrided_atb_total_reorder_4D=np.empty( (subcolum_num,time_num,Nglevels,col_num))
    Regrided_atb_mol_reorder_4D=np.empty( (subcolum_num,time_num,Nglevels,col_num))


    for kk in np.arange(col_num):  

        zfull=z_full_km_3D[:,:,kk]
        zhalf=z_half_km_3D[:,:,kk]


        y_input_atbtot=np.transpose(atb_total_4D_num[:,:,:,kk],axes=(1,0,2))  # loop through columns
        y_input_abtmol=np.transpose(atb_mol_4D_num[:,:,:,kk],axes=(1,0,2))  # loop through columns

        Regrided_atb_total_3D=COSP_CHANGE_VERTICAL_GRID(Npoints,Ncolumns,Nlevels,zfull,zhalf,y_input_atbtot,Nglevels,newgrid_bot,newgrid_top,lunits=False)
        Regrided_atb_mol_3D=COSP_CHANGE_VERTICAL_GRID(Npoints,Ncolumns,Nlevels,zfull,zhalf,y_input_abtmol,Nglevels,newgrid_bot,newgrid_top,lunits=False)


        # need to covert to right dimension order
        Regrided_atb_total_reorder_4D[:,:,:,kk]=np.transpose(Regrided_atb_total_3D,axes=(1,0,2))  # loop through columns
        Regrided_atb_mol_reorder_4D[:,:,:,kk]=np.transpose(Regrided_atb_mol_3D,axes=(1,0,2))  # loop through columns


    loc_atbtot=np.where(Regrided_atb_total_reorder_4D==R_UNDEF)
    Regrided_atb_total_reorder_4D[loc_atbtot]=np.nan

    loc_atbmol=np.where(Regrided_atb_mol_reorder_4D==R_UNDEF)
    Regrided_atb_mol_reorder_4D[loc_atbmol]=np.nan

    SR_4D=Regrided_atb_total_reorder_4D/Regrided_atb_mol_reorder_4D

    return SR_4D
    

def COSP_CHANGE_VERTICAL_GRID(Npoints,Ncolumns,Nlevels,zfull,zhalf,y,Nglevels,newgrid_bot,newgrid_top,lunits=False):

    
    r = np.zeros([Npoints, Ncolumns, Nglevels]) # (1236, 20 , 40)

    R_UNDEF      = -1.0E30   #   Missing value
    R_GROUND     = -1.0E20   #   Flag for below ground results

    oldgrid_top=np.zeros(Nlevels)
    
    
    
    for i in np.arange(Npoints): # loop through each grid
        
        # Calculate tops and bottoms of new and old grids
        oldgrid_bot = zhalf[i,:].copy()
        oldgrid_top[:Nlevels-1] = oldgrid_bot[1:].copy()
        oldgrid_top[Nlevels-1] = zfull[i,Nlevels-1] +  zfull[i,Nlevels-1] - zhalf[i,Nlevels-1] # Top level symmetric


        l = -1 # Index of level in the old grid



        for k in np.arange(Nglevels):   # Loop over levels in the new grid
            
            Nw = 0  # Number of weigths
            wt = 0. # Sum of weights
            # Loop over levels in the old grid and accumulate total for weighted average
            # change do loop in Fortran to while loop in Python
            while l < Nlevels:

                
                l = l + 1  # return to while loop
                
                
                w = 0.0 # Initialise weight to 0

                 # Distances between edges of both grids
                dbb = oldgrid_bot[l] - newgrid_bot[k]
                dtb = oldgrid_top[l] - newgrid_bot[k]
                dbt = oldgrid_bot[l] - newgrid_top[k]
                dtt = oldgrid_top[l] - newgrid_top[k]

                if dbt >= 0.:
#                    print(l, dbt)
                    break   # Do next level in the new grid (k=k+1)
                if dtb > 0.:
                    if dbb <= 0.0:
                        if dtt <= 0:
                            w = dtb.copy()
                        else:
                            w = newgrid_top[k] - newgrid_bot[k]

                    else:
                        if dtt <= 0:
                            w = oldgrid_top[l] - oldgrid_bot[l]
                        else:
                            w = -dbt

                   # If layers overlap (w!=0), then accumulate
                    if w != 0.0:
                        Nw = Nw + 1
                        wt = wt + w

                        for j in np.arange(Ncolumns):
                            if lunits:  # True
                                if (y[i,j,l] != R_UNDEF):
                                    yp = 10.**(y[i,j,l]/10.)
                                else:
                                    yp = 0.
                            else:
                                yp = y[i,j,l] 

                            r[i,j,k] = r[i,j,k] + w*yp
                            
#                            print(r[i,j,k],w,10.*np.log10(r[i,j,k]) )


                
#            print('A',l)
            
            l = l - 2
            if l < 1:
                l = 0

#            print('B',l)


            # Calculate average in new grid
            if Nw > 0:
                for j in np.arange(Ncolumns):
                    r[i,j,k] = r[i,j,k]/wt
                    
#                    if j == 10: 
#                        print(i,k,r[i,j,k], 10.*np.log10(r[i,j,k]))



    for k in np.arange(Nglevels):
        for j in np.arange(Ncolumns):
            for i in np.arange(Npoints):
#            for i in np.arange(1)+480:
        
                if newgrid_top[k] > zhalf[i,0]:  # Level above model bottom level
                    if lunits:    
                        if r[i,j,k] <= 0.0:
                            r[i,j,k] = R_UNDEF
                        else:
                            r[i,j,k] = 10.*np.log10(r[i,j,k])  
                            

                else: # Level below surface
                    r[i,j,k] = R_GROUND

        
                    
    return r





def get_regridded_ze_att_tot(Ze_att_total_4D,z_full_km_3D,z_half_km_3D,subcolum_num,time_num,Nglevels,col_num,newgrid_bot,newgrid_top,Ncolumns,Npoints,Nlevels):
    
    R_UNDEF      = -1.0E30   #   Missing value

    a=copy.deepcopy(Ze_att_total_4D)
    Ze_att_total_4D_num=np.nan_to_num(a, nan=R_UNDEF)

    Regrided_Ze_att_tot_reorder_4D=np.empty( (subcolum_num,time_num,Nglevels,col_num))

    for kk in np.arange(col_num):  

        zfull=z_full_km_3D[:,:,kk]
        zhalf=z_half_km_3D[:,:,kk]


        y_input_zeatttot=np.transpose(Ze_att_total_4D_num[:,:,:,kk],axes=(1,0,2))  # loop through columns

        Regrided_ze_att_total_3D=COSP_CHANGE_VERTICAL_GRID(Npoints,Ncolumns,Nlevels,zfull,zhalf,y_input_zeatttot,Nglevels,newgrid_bot,newgrid_top,lunits=False)

        # need to covert to right dimension order
        Regrided_Ze_att_tot_reorder_4D[:,:,:,kk]=np.transpose(Regrided_ze_att_total_3D,axes=(1,0,2))  # loop through columns

    loc_ze=np.where(Regrided_Ze_att_tot_reorder_4D==R_UNDEF)
    Regrided_Ze_att_tot_reorder_4D[loc_ze]=np.nan

    return Regrided_Ze_att_tot_reorder_4D




def calculate_lidar_CF(SR_4D,time_num,Nglevels,col_num):
    
    S_cld=5.
    s_att=0.01    

    CF_3D=np.empty((time_num,Nglevels,col_num))*np.nan
    #Cloud detection at subgrid-scale
    cldy_pixels=np.empty_like(CF_3D)  #(50, 24, 40, 3)
    # Number of usefull sub-columns:
    srok_pixels=np.empty_like(CF_3D)  #(50, 24, 40, 3)

    for i in np.arange(time_num):
        for j in np.arange(Nglevels):
            for k in np.arange(col_num):

                cldy_pixels[i,j,k]=np.where( SR_4D[:,i,j,k] > S_cld)[0].shape[0]
                srok_pixels[i,j,k]=np.where( SR_4D[:,i,j,k] > s_att)[0].shape[0] 

                if srok_pixels[i,j,k]>0:
                    CF_3D[i,j,k]=cldy_pixels[i,j,k]*1.0/srok_pixels[i,j,k]

    return  CF_3D  


#https://github.com/CFMIP/COSPv2.0/blob/master/src/simulator/quickbeam/quickbeam.F90#L293

def cal_cfad_radar_40levels(nsubcolumn,levStat_km,Ze_EDGES,a):
    
    
    cfaddbz94_space_cal_sub=np.empty( ( ( (len(levStat_km)) ,(len(Ze_EDGES)-1) ) ) )*np.nan

    for i in np.arange(len(levStat_km)): 
        
        a1=a[:,i]  
        
        loc0=np.where( (a1>= Ze_EDGES[0] ) & (a1<Ze_EDGES[-1]) )
        
        if loc0[0].shape[0] > 0:
        
            for j in np.arange(len(Ze_EDGES)-1):   

                loc1=np.where( (a1>= Ze_EDGES[j] ) & (a1<Ze_EDGES[j+1]) ) 
            
                cfaddbz94_space_cal_sub[i,j]=len(loc1[0])*1.0/nsubcolumn
                
                    
    return cfaddbz94_space_cal_sub

def get_cfaddBZ(Ze_EDGES,newgrid_mid,Npoints,Ncolumns,Regrided_Ze_att_tot_reorder_4D,col_index):


    # height
    levStat_km=copy.deepcopy(newgrid_mid)
    levStat_km_add0=np.arange(len(levStat_km)+1)*0.48

    # signal
    cfaddbz35_cal=np.empty( (len(levStat_km) ,(len(Ze_EDGES)-1),  Npoints   ) )*np.nan  # levels
    Ze_forCFAD=Regrided_Ze_att_tot_reorder_4D[:,:,:,col_index]

    for kkk in np.arange(Npoints):

        dbz_2d=Ze_forCFAD[:,kkk,:]   
        cfaddbz35_cal[:,:,kkk]=cal_cfad_radar_40levels(Ncolumns,levStat_km,Ze_EDGES,dbz_2d)

    cfaddbz35_cal_alltime=np.nansum(cfaddbz35_cal,axis=2)/Npoints   

    return cfaddbz35_cal_alltime



def get_cfad_SR(SR_EDGES,newgrid_mid,Npoints,Ncolumns,SR_4D,col_index):

    
    # how to use
    # https://github.com/CFMIP/COSPv2.0/blob/master/src/simulator/actsim/lidar_simulator.F90#L457
    # https://github.com/CFMIP/COSPv2.0/blob/master/src/simulator/actsim/lidar_simulator.F90#L287  #ncol,    & ! Number of subcolumns
    
    # height
    levStat_km=copy.deepcopy(newgrid_mid)
    levStat_km_add0=np.arange(len(levStat_km)+1)*0.48


    cfadSR_cal=np.empty( (len(levStat_km) ,(len(SR_EDGES)-1),  Npoints   ) )*np.nan  # levels

    SR_forCFAD=SR_4D[:,:,:,col_index]

    for kkk in np.arange(Npoints):
        SR_2d=SR_forCFAD[:,kkk,:]  
        cfadSR_cal[:,:,kkk]=cal_cfad_radar_40levels(Ncolumns,levStat_km,SR_EDGES,SR_2d)

    cfadSR_cal_alltime=np.nansum(cfadSR_cal,axis=2)/Npoints
    
    return cfadSR_cal_alltime


