# EngDes2020_Dyn
Progress from the 2018, 2019, 2020 teams of UChicago MENG students working on the Spray Dynamics team for the Engineering Design course sequence (Capstone Project).

This branch (**parallelization**) is the GPU implementation version of the developed algorithms by the 2020-2021 Dynamics Team. We utilize an AMD Navi 10 card (RX 5700XT) to parallelize several of the algorithms that require many matrix operations - this is done across the . The code is capable of running on other heterogenous systems given some slight modifications to optimize to the system being used. 



Heterogenous System Utilized (Workstation):
- OS: Ubuntu 18.04 LTS dual boot installation (Linux distro)
1. Motherboard: ASUS TUF GAMING X570-PLUS (AMD AM4 Socket)
2. CPU: AMD Ryzen 7 3700X - overclocked (OC) @ 4.03 GHz
3. GPU: AMD Radeon RX 5700XT ASRock Challenger OC - OC @ ? (~1800 MHz?)
4. RAM: Corsair 64 GB DDR4 - 3200 MHz CL16
5. SSD: 1TB Rocket 4 PLUS NVMe 4.0 Gen4 PCIe M.2
  - OS installed on 350 GB partition
  - This physical drive is not the original OS drive, therefore, bandwidth speed/size might be limited
