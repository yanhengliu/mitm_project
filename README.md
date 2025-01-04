### Group Members

- Yanheng Liu
- Ghazaleh Kharadpoor

### Run

To run this project, follow these steps:

1. **Ensure the Project Directory Structure**  
   Verify that you are in the `mitm_project` root directory, and ensure the `src` folder contains all the required source files (`dictionary.c`, `distributed_dict.c`, `speck.c`, `utils.c`, `main.c`) and their corresponding headers.

2. **Compile the Project**  
   Use the provided `makefile` to compile the code:

   ```bash
   make
   ```

   This will generate the executable file named `mitm` in the bin directory.

3. **Execute the Program**  
   Run the program by exapme executing, you could replace the parameters as you want:

   ```bash
   mpirun -np 4 bin/mitm --n 16 --C0 df859b389989b79b --C1 87168f99c5b83789
   ```

   ```bash
   mpirun -np 4 bin/mitm --n 30 --C0 12392f511a159c4f --C1 c840cb1a5527641d
   ```
   Or running on the Gird5000
   ```bash
   mpiexec --n 30 --hostfile $OAR_NODEFILE bin/mitm --n 33 --C0 1ba377d7cab89927 --C1 a5edede78a7dbe3d
   ```
4. **Optional: Clean Compiled Files**  
   To clean up object files and the generated executable, run:
   ```bash
   make clean
   ```
