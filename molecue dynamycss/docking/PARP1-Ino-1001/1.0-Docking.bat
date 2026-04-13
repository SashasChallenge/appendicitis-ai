@echo off
for %%f in (Ino_1001.pdbqt) do (
echo Processing %%f
mkdir %%f_result
"C:\Program Files (x86)\The Scripps Research Institute\Vina\vina.exe" --config conf.txt --ligand %%f --out %%f_result\%%f_out.pdb --log %%f_result\%%f_log.txt
)
echo 任务完成！ 
pause&exit