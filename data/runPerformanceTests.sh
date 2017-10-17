# Basic SW performance tests
for i in `ls data/*fasta_*.fasta`; do echo $i;	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPU.log.txt 	--device_type=CPU --platform_name=Intel --framework=OpenCL -o speed.txt; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPUSingle.log.txt 	--device_type=CPU --number_of_compute_units=1 --platform_name=Intel --framework=OpenCL -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPUocl.log.txt 	--device_type=CPU --platform_name=Intel --framework=OpenCLforceGPU -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPUoclSingle.log.txt 	--device_type=CPU --number_of_compute_units=1 --platform_name=Intel --framework=OpenCLforceGPU -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.GPUOpenCL.log.txt 	--device_type=GPU --platform_name=NVIDIA --framework=OpenCL -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.GPUCUDA.log.txt 	--device_type=GPU --platform_name=NVIDIA --framework=CUDA -o speed.txt ; done

# Gap extension
for i in `ls data/*fasta_*.fasta`; do echo $i;	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPU.GE.log.txt 	--device_type=CPU --platform_name=Intel --framework=OpenCL -g -1 -o speed.txt; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPUSingle.GE.log.txt 	--device_type=CPU --number_of_compute_units=1 --platform_name=Intel --framework=OpenCL -g -1 -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPUocl.GE.log.txt 	--device_type=CPU --platform_name=Intel --framework=OpenCLforceGPU -g -1 -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.CPUoclSingle.GE.log.txt 	--device_type=CPU --number_of_compute_units=1 --platform_name=Intel --framework=OpenCLforceGPU -g -1 -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.GPUOpenCL.GE.log.txt 	--device_type=GPU --platform_name=NVIDIA --framework=OpenCL -g -1 -o speed.txt ; done
for i in `ls data/*fasta_*.fasta`; do echo $i; 	python pypaswas.py --loglevel=info -M BLOSUM80 data/canisLupusAnkyrin.fasta $i -L $i.GPUCUDA.GE.log.txt 	--device_type=GPU --platform_name=NVIDIA --framework=CUDA -g -1 -o speed.txt ; done
