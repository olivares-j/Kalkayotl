# rsync -av -r -e ssh --exclude='*.nc' --exclude='*.h5' dance13:/raid/jromero/Kalkayotl/Synthetic/Gaussian_* /home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/
# rsync -av -r -e ssh --exclude='*.nc' --exclude='*.h5' dance14:/home/jromero/Kalkayotl/Synthetic/StudentT_* /home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/
#rsync -av -r -e ssh --exclude='*.nc' --exclude='*.h5' dance14:/home/jromero/Kalkayotl/Synthetic/GMM_joint /home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/
rsync -av -r -e ssh --exclude='*.nc' --exclude='*.h5' dance13:/raid/jromero/Kalkayotl/Synthetic/GMM_joint /home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/