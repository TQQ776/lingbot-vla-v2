import serial
import time
import struct
import threading

# 夹爪位置定义
GRIPPER_50MINI_POS = 500
GRIPPER_100MINI_POS = 200
GRIPPER_MAX_POS = 2048

class BusServo:
    # 指令定义
    INST_PING = 0x01
    INST_READ_DATA = 0x02
    INST_WRITE_DATA = 0x03
    INST_REG_WRITE = 0x04
    INST_ACTION = 0x05
    INST_RESET = 0x06
    INST_SYNC_READ = 0x82
    INST_SYNC_WRITE = 0x83
    
    # 特殊ID
    ID_BROADCAST = 0xFE

     # --- 夹爪校准数据表 ---
    # 格式: (张开毫米数, 对应的舵机数值)
    # 注意：必须按照毫米数从小到大排序
    GRIPPER_CALIBRATION = {
        # 你提供的 100mm 夹爪实测数据
        "100mm": [
            (0, 2048),    # 闭合
            (10, 1700),
            (20, 1450),
            (30, 1280),
            (40, 1120),
            (50, 980),
            (60, 850),
            (70, 720),
            (80, 590),
            (90, 440),
            (100, 198)    # 最大张开
        ],
        # 50mm 的暂时保留为线性估算或可以在此补充实测数据
        "50mm": [
            (0, 2048),    # 假设闭合是 2048
            (5, 1900),
            (10, 1775),  
            (15, 1645),
            (20, 1530),
            (25, 1430),
            (30, 1320),
            (35, 1215),
            (40, 1085),
            (45, 960),
            (50, 798)    
        ]
    }
    
    def __init__(self, port, baudrate=115200, timeout=0.1, verbose=True):
        """
        初始化串口连接
        :param port: 串口号
        :param baudrate: 波特率
        :param timeout: 读取超时时间
        :param verbose: 是否打印收发数据日志 (默认开启)
        """
        self.serial = serial.Serial(port, baudrate, timeout=timeout)
        self.verbose = verbose
        self._impedance_running = False  
        
    def close(self):
        if self.serial.is_open:
            self.serial.close()

    def _log(self, direction, data):
        """
        打印十六进制日志
        :param direction: "TX" (发送) 或 "RX" (接收)
        :param data: 字节数据 (list 或 bytearray)
        """
        if self.verbose and data:
            hex_str = ' '.join(f'{b:02X}' for b in data)
            print(f"[{direction}] {hex_str}")

    def _calc_checksum(self, id, length, instruction, params):
        total = id + length + instruction + sum(params)
        return (~total) & 0xFF

    def _send_packet(self, id, instruction, params):
        """
        构建并发送数据包
        """
        length = len(params) + 2
        checksum = self._calc_checksum(id, length, instruction, params)
        
        packet = [0xFF, 0xFF, id, length, instruction] + params + [checksum]
        packet_bytes = bytearray(packet)
        
        # 打印发送日志
        # self._log("TX", packet_bytes)
        
        # 清空输入缓冲区并发送
        self.serial.reset_input_buffer()
        self.serial.write(packet_bytes)
        
        return packet_bytes

    def _receive_packet(self):
        """
        接收应答包
        """
        raw_bytes = bytearray() # 用于存储接收到的所有字节以供打印

        try:
            # 读取包头 (2字节)
            header = self.serial.read(2)
            if not header:
                return None, "Timeout (No Header)"
            raw_bytes.extend(header)
            
            if header != b'\xFF\xFF':
                self._log("RX", raw_bytes) # 打印错误的包
                return None, f"Header Error: {header.hex()}"
            
            # 读取 ID (1字节)
            id_bytes = self.serial.read(1)
            if not id_bytes: 
                self._log("RX", raw_bytes)
                return None, "Timeout (No ID)"
            raw_bytes.extend(id_bytes)
            resp_id = id_bytes[0]
            
            # 读取长度 (1字节)
            len_bytes = self.serial.read(1)
            if not len_bytes: 
                self._log("RX", raw_bytes)
                return None, "Timeout (No Length)"
            raw_bytes.extend(len_bytes)
            length = len_bytes[0]
            
            # 读取剩余数据 (Length 字节)
            # Length = Error(1) + Params(N) + Checksum(1)
            remaining_data = self.serial.read(length)
            raw_bytes.extend(remaining_data)
            
            # 打印完整的接收日志
            # self._log("RX", raw_bytes)

            if len(remaining_data) != length:
                return None, f"Packet Incomplete. Expected {length}, got {len(remaining_data)}"
                
            error = remaining_data[0]
            params = list(remaining_data[1:-1])
            received_checksum = remaining_data[-1]
            
            # 校验和验证
            calc_sum = self._calc_checksum(resp_id, length, error, params)
            
            if calc_sum != received_checksum:
                return None, f"Checksum Error (Calc: {hex(calc_sum)}, Recv: {hex(received_checksum)})"
                
            return params, error
            
        except Exception as e:
            self._log("RX", raw_bytes)
            return None, str(e)

    # --- 功能接口 ---

    def ping(self, servo_id):
        """ 查询状态 """
        self._send_packet(servo_id, self.INST_PING, [])
        data, err = self._receive_packet()
        return err

    def read_data(self, servo_id, address, length):
        """ 读内存 """
        self._send_packet(servo_id, self.INST_READ_DATA, [address, length])
        data, err = self._receive_packet()
        if data and len(data) == length:
            return data
        return None
    
    def write_data(self, servo_id, address, values):
        """
        1.3.3 写指令 WRITE DATA
        :param values: 字节列表 [byte1, byte2...]
        """
        params = [address] + values
        self._send_packet(servo_id, self.INST_WRITE_DATA, params)
        # 广播ID (0xFE/254) 不返回应答
        if servo_id != self.ID_BROADCAST:
            return self._receive_packet()
        return None, 0

    def reg_write(self, servo_id, address, values):
        """
        1.3.4 异步写指令 REG WRITE
        """
        params = [address] + values
        self._send_packet(servo_id, self.INST_REG_WRITE, params)
        if servo_id != self.ID_BROADCAST:
            return self._receive_packet()
        return None, 0

    def action(self, servo_id=ID_BROADCAST):
        """
        1.3.5 执行异步写指令 ACTION
        通常使用广播ID触发所有舵机行动
        """
        self._send_packet(servo_id, self.INST_ACTION, [])
        # ACTION 不返回数据

    def sync_write(self, address, data_len, servo_data):
        """
        1.3.6 同步写指令 SYNC WRITE
        :param address: 写入的首地址
        :param data_len: 每个舵机写入的数据长度
        :param servo_data: 列表，格式为 [[id1, val1, val2...], [id2, val1, val2...]]
        """
        params = [address, data_len]
        for item in servo_data:
            params.extend(item) # Flatten list: ID, Data1, Data2...
            
        self._send_packet(self.ID_BROADCAST, self.INST_SYNC_WRITE, params)

    def move_servo(self, servo_id, position, torque, speed):
        """ 
        控制舵机转动 
        Addr: 0x2A ( 位置 2byte, 力矩 2byte, 速度 2byte)
        """
        params = list(struct.pack('<BHHH', servo_id, position, torque, speed))
        self.sync_write(0x2A, 6, [params])
    
    # --- 应用接口 ---

    def set_servo_id(self, old_id, new_id):
        """ 设置舵机ID (Addr: 0x05, Len: 1) """
        self.write_data(old_id, 0x05, [new_id])

    def set_baudrate(self, servo_id, baudrate):
        """ 设置波特率 (Addr: 0x06, Len: 1) """
        # 波特率计算公式: Baudrate = 2000000 / (X + 1)
        x = int(2000000 / baudrate) - 1
        self.write_data(servo_id, 0x06, [x])

    def set_middle_position(self, servo_id):
        """ 设置当前位置为中位点 (Addr: 0x28, ) """
        checksum = servo_id + 0x04 + 0x0B + 0x00 + 0x08
        checksum = (~checksum) & 0xFF
        packet = [0xFF, 0xFF, servo_id, 0x04,0x0B,0x00,0x08,checksum]
        # 清空输入缓冲区并发送
        self.serial.reset_input_buffer()
        self.serial.write(bytearray(packet))

    def set_servo_torque_enable(self, servo_id, enable=True):
        """ 设置扭矩开关 (Addr: 0x28, Len: 1) """
        val = 1 if enable else 0
        self.write_data(servo_id, 0x28, [val])

    def get_position(self, servo_id):
        """ 读取位置 (Addr: 0x38, Len: 2) """
        data = self.read_data(servo_id, 0x38, 2)
        if data:
            return struct.unpack('<h', bytearray(data))[0]
        return None

    def read_sensor_data(self, servo_id):
        """
        读取传感器的综合状态 (地址 0x38, 长度 8)
        包含: 位置(2B), 速度(2B), 负载(2B), 电压(1B), 温度(1B)
        """
        # 根据协议 Page 8 例8: Start Address 0x38, Length 8
        data = self.read_data(servo_id, 0x38, 8)
        
        if data and len(data) == 8:
            # 解析数据：pos, speed 为有符号 16 位，load 为无符号 16 位，volt/temp 为 1 字节
            pos, speed, raw_load, volt, temp = struct.unpack('<hhHBB', bytearray(data))

            load = int(raw_load)
            load = load - 1024

            return {
                "id": servo_id,
                "position": pos,
                "speed": speed,
                "load": load,             # 当前负载（int 类型）
                "voltage": volt,          # 当前电压 (通常单位是 0.1V)
                "temperature": temp       # 当前温度 (单位 ℃)
            }
        return None

    def _interpolate(self, value, table):
        """
        线性插值计算
        :param value: 目标宽度 (mm)
        :param table: 校准表 list of (mm, servo_val)
        """
        # 边界处理
        if value <= table[0][0]:
            return table[0][1]
        if value >= table[-1][0]:
            return table[-1][1]

        # 寻找 value 所在的区间
        for i in range(len(table) - 1):
            x0, y0 = table[i]
            x1, y1 = table[i+1]

            if x0 <= value <= x1:
                # 线性插值公式
                ratio = (value - x0) / (x1 - x0)
                result_pos = y0 + ratio * (y1 - y0)
                return int(result_pos)
        
        return table[-1][1]

    def set_gripper_position(self, servo_id, gripper_type, position_mm, torque, speed):
        """
        设置夹爪位置 (基于实测数据插值)
        :param servo_id: 舵机ID
        :param gripper_type: "50mm" 或 "100mm"
        :param position_mm: 目标位置 (mm)
        :param torque: 力矩 (0-1000)
        :param speed: 速度 (力控版本夹爪范围0-150，非力控版本范围0-3000)
        """
        if gripper_type not in self.GRIPPER_CALIBRATION:
            print(f"Error: Unknown gripper type '{gripper_type}'")
            return

        # 获取对应的校准表
        calib_table = self.GRIPPER_CALIBRATION[gripper_type]
        
        # 计算舵机数值
        target_pos = self._interpolate(position_mm, calib_table)
        
        # 安全限位钳制 (防止插值算出超出舵机物理极限的值)
        target_pos = max(0, min(4096, target_pos))

        if self.verbose:
            print(f"--> Gripper Control: Type={gripper_type}, Target={position_mm}mm, ServoPos={target_pos}")

        self.move_servo(servo_id, target_pos, torque, speed)

    def _servo_pos_to_mm(self, servo_pos, gripper_type):
        """
        反向插值：将舵机位置值转换为夹爪开口毫米数
        :param servo_pos: 当前舵机位置值
        :param gripper_type: "50mm" 或 "100mm"
        :return: 对应的开口毫米数
        """
        if gripper_type not in self.GRIPPER_CALIBRATION:
            return None
        table = self.GRIPPER_CALIBRATION[gripper_type]
        # 校准表中 servo_val 是递减的 (mm增大时servo_val减小)
        # 边界处理
        if servo_pos >= table[0][1]:   # 大于闭合值
            return table[0][0]
        if servo_pos <= table[-1][1]:  # 小于最大张开值
            return table[-1][0]
        # 反向查找区间 (servo_val 递减)
        for i in range(len(table) - 1):
            mm0, sv0 = table[i]
            mm1, sv1 = table[i + 1]
            if sv1 <= servo_pos <= sv0:
                ratio = (sv0 - servo_pos) / (sv0 - sv1)
                return mm0 + ratio * (mm1 - mm0)
        return table[-1][0]

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # Windows: 'COM3', Linux: '/dev/ttyUSB0', Mac: '/dev/tty.usbserial-xxx'
    servo = BusServo(port='COM5', baudrate=1000000, verbose=True)

    try:
            print("\n--- TEST 1: PING ---")
            # 查询舵机状态
            err = servo.ping(1)
            if err == 0:
                print("Ping Success!")
            else:
                print(f"Ping Failed or Error Code: {err}")

            # 设置当前位点为中位点
            # servo.set_middle_position(254)
            # time.sleep(0.05)

            print("\n--- TEST 2: READ SENSOR DATA ---")
            # 读取传感器数据
            sensor_data = servo.read_sensor_data(1)
            if sensor_data:
                print("Sensor Data:", sensor_data)
            else:
                print("Failed to read sensor data.")
            
            print("\n--- TEST 3: SET GRIPPER POSITION ---")
            # 控制夹爪到 100mm 开口位置，速度 2000
            servo.set_gripper_position(1, "100mm", 100, 0, 2000)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        servo.close()
    