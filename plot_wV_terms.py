import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


# plot only measurement

meas_file_versus = '/disks/shear16/herle/correlations1/correlations_voids_versus_20_30Mpc_L1000N1800_HYDRO_FIDUCIAL_0.0.h5'
df_versus = pd.read_hdf(meas_file_versus, key='a')
rp_meas_versus = df_versus['wgp_rp'].values
wvp_meas_versus = df_versus['wgp_xip'].values
err_meas_versus = df_versus['wgp_std_from_cov'].values

# meas_file_zobov = '/disks/shear16/herle/correlations1/correlations_voids_zobov_L1000N1800_HYDRO_FIDUCIAL_0.0.h5'
# df_zobov = pd.read_hdf(meas_file_zobov, key='a')
# rp_meas_zobov = df_zobov['wgp_rp'].values
# wvp_meas_zobov = df_zobov['wgp_xip'].values
# err_meas_zobov = df_zobov['wgp_std_from_cov'].values
    

plt.figure(figsize=(6, 5))



plt.errorbar(
    rp_meas_versus,
    rp_meas_versus*wvp_meas_versus,
    yerr=rp_meas_versus*err_meas_versus,
    fmt='o',
    capsize=3,
    label='Versus'
)

plt.ylim(-15, 15)

plt.xscale('log')
plt.xlabel(r'$r_p$ [Mpc]')
plt.ylabel(r'$r_p w_{\mathrm{V}+} [\mathrm{Mpc}^2]$')

ax = plt.gca()



plt.hlines(0, xmin=1, xmax=800, colors='k', linestyles='dotted')

plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('/disks/shear16/herle/code/plots/versus_20_30Mpc.png')

# base_folder = '/disks/shear16/herle/models/voids/pimax100/redshift0/'


# A1_v = 0.04
# b_v = 1.5
# h = 0.681
# scaling = 1.0/(np.pi**2)


# wVII = np.load(base_folder+'/wgg_1.npy')
# wVII *= scaling
# wVII /=h

# wVII *= A1_v*b_v

# wVI = np.load(base_folder+'/wgg_2.npy')
# wVI *= scaling
# wVI *= A1_v
# wVI /= h

# rp = np.loadtxt(base_folder+'/w_gp_z0.0.txt')[:, 0]

# # masks
# mask_VI = (rp >= 6) & (rp <= 35)
# mask_VII = (rp >= 60) & (rp <= 200)


# run = 'L1000N1800'
# redshift = '0.0'

# meas_file_disperse = '/disks/shear16/herle/correlations1/correlations_voids_versus_10_20Mpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.h5'
# df_disperse = pd.read_hdf(meas_file_disperse, key='a')
# rp_meas_disperse = df_disperse['wgp_rp'].values
# wvp_meas_disperse = df_disperse['wgp_xip'].values
# err_meas_disperse = df_disperse['wgp_std_from_cov'].values



# # meas_file_voxel = '/disks/shear16/herle/correlations1/correlations_voids_voxel_10_30Mpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.h5'
# # df_voxel = pd.read_hdf(meas_file_voxel, key='a')
# # rp_meas_voxel = df_voxel['wgp_rp'].values
# # wvp_meas_voxel = df_voxel['wgp_xip'].values
# # err_meas_voxel = df_voxel['wgp_std_from_cov'].values

# # meas_file_zobov = '/disks/shear16/herle/correlations1/correlations_voids_zobov_10_30Mpc_'+run+'_HYDRO_FIDUCIAL_'+redshift+'.h5'
# # df_zobov = pd.read_hdf(meas_file_zobov, key='a')
# # rp_meas_zobov = df_zobov['wgp_rp'].values
# # wvp_meas_zobov = df_zobov['wgp_xip'].values
# # err_meas_zobov = df_zobov['wgp_std_from_cov'].values


# plt.figure(figsize=(6, 5))


# plt.plot(rp, rp*wVII, alpha=0.2, ls='--')
# plt.plot(rp, rp*wVI,  alpha=0.2, ls='--')

# plt.errorbar(
#     rp_meas_disperse,
#     rp_meas_disperse*wvp_meas_disperse,
#     yerr=rp_meas_disperse*err_meas_disperse,
#     fmt='o',
#     capsize=3,
#     label='Disperse'
# )

# # plt.errorbar(
# #     rp_meas_voxel,
# #     rp_meas_voxel*wvp_meas_voxel,
# #     yerr=rp_meas_voxel*err_meas_voxel,
# #     fmt='s',
# #     capsize=3,
# #     label='Revolver (Voxel)'
# # )

# # plt.errorbar(
# #     rp_meas_zobov,
# #     rp_meas_zobov*wvp_meas_zobov,
# #     yerr=rp_meas_zobov*err_meas_zobov,
# #     fmt='^',
# #     capsize=3,
# #     label='Revolver (Zobov)'
# # )

# plt.ylim(-15, 15)

# plt.xscale('log')
# plt.xlabel(r'$r_p$ [Mpc]')
# plt.ylabel(r'$r_p w_{\mathrm{V}+} [\mathrm{Mpc}^2]$')

# ax = plt.gca()
# correct_tick_marks(ax)

# line_VII, = plt.plot(rp[mask_VII], rp[mask_VII]*wVII[mask_VII], label='2-void')
# line_VI,  = plt.plot(rp[mask_VI],  rp[mask_VI]*wVI[mask_VI],   label='1-void')
# plt.axvspan(60, 200, color=line_VII.get_color(), alpha=0.08, zorder=0)
# plt.axvspan(6,  35,  color=line_VI.get_color(),  alpha=0.08, zorder=0)


# plt.hlines(0, xmin=1, xmax=300, colors='k', linestyles='dotted')

# plt.legend(frameon=False)

# plt.tight_layout()
# plt.savefig('/disks/shear16/herle/code/plots/versus_10_20Mpc.png')