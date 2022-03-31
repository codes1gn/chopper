extern crate backend_vulkan as concrete_backend;

use std::collections::HashMap;

use crate::base::errors::EmptyCmdBufferError;
use crate::base::errors::RuntimeStatusError;
use crate::base::*;
use crate::instruction::OpCode;

use crate::buffer_view::*;
use crate::instance::*;
use crate::session::*;

#[derive(Debug)]
pub struct VM<'a> {
    registers: [i32; 32],
    command_buffer: Vec<u8>,
    // use usize since this is decided by the arch of computer
    // 64/32 bits, it is equivalent to unsigned long long in C/C++
    // for a program counter, must use this to enumerate the reals.
    program_counter: usize,
    // TODO to bring device instance into interpreter, may need to impl Default
    // to allow new without explicit value of Session, thus not borrow a moved
    // value -> device instance
    session: Session<'a>,
    // data_buffer_f32: Vec<DataView<concrete_backend::Backend, f32>>,
    data_buffer_f32: HashMap<usize, DataView<concrete_backend::Backend, f32>>,
    // data_buffer_i32: Vec<DataView<concrete_backend::Backend, i32>>,
    data_buffer_i32: HashMap<usize, DataView<concrete_backend::Backend, i32>>,
}

impl<'a> VM<'a> {
    pub fn new(dinstance: &'a DeviceInstance) -> VM<'a> {
        let mut session = Session::new(dinstance);
        session.init();
        VM {
            registers: [0; 32],
            program_counter: 0,
            command_buffer: vec![],
            session: session,
            data_buffer_f32: HashMap::new(),
            data_buffer_i32: HashMap::new(),
        }
    }

    fn fetch_instruction(&mut self) -> Result<OpCode, EmptyCmdBufferError> {
        if self.program_counter > self.command_buffer.len() {
            return Err(EmptyCmdBufferError);
        }
        let opcode = OpCode::from(self.command_buffer[self.program_counter]);
        self.program_counter += 1;
        Ok(opcode)
    }

    fn get_next_byte(&mut self) -> u8 {
        let _cmd_buffer = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        _cmd_buffer
    }

    fn get_next_two_bytes(&mut self) -> u16 {
        let _cmd_buffer = ((self.command_buffer[self.program_counter] as u16) << 8)
            | self.command_buffer[self.program_counter + 1] as u16;
        self.program_counter += 2;
        _cmd_buffer
    }

    fn get_next_four_bytes(&mut self) -> [u8; 4] {
        let mut ret_bytes = [0; 4];
        ret_bytes[0] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[1] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[2] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes[3] = self.command_buffer[self.program_counter];
        self.program_counter += 1;
        ret_bytes
    }

    // TODO may replace status with a enum
    fn step(&mut self) -> Result<u8, RuntimeStatusError> {
        println!("start execute step");
        match self.fetch_instruction().unwrap() {
            OpCode::HALT => {
                println!("halt to exit");
                // TODO move to base::const, use 1 as halt status
                Ok(1)
            }
            OpCode::ILLEGAL => {
                println!("Illegal instruction found");
                Err(RuntimeStatusError)
            }
            OpCode::LOAD => {
                let register_id = self.get_next_byte() as usize;
                let operand = self.get_next_two_bytes() as u16;
                // note the registers is defaultly i32s
                self.registers[register_id] = operand as i32;
                // TODO change return of error code as error enum
                // TODO change into verbose string
                Ok(0)
            }
            OpCode::ADDI32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                println!("{:?}", lhs_dataview);
                println!("{:?}", rhs_dataview);
                let opcode = OpCode::ADDI32;
                let outs = self
                    .session
                    .benchmark_run::<i32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::ADDF32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::ADDF32;
                let outs = self
                    .session
                    .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::SUBI32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::SUBI32;
                let outs = self
                    .session
                    .benchmark_run::<i32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::SUBF32 => {
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::SUBF32;
                let outs = self
                    .session
                    .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::MULI32 => {
                // TODO merge logics together and pass the OpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::MULI32;
                let outs = self
                    .session
                    .benchmark_run::<i32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::MULF32 => {
                // TODO merge logics together and pass the OpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::MULF32;
                let outs = self
                    .session
                    .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::FLOORDIVI32 => {
                // TODO merge logics together and pass the OpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_i32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_i32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::FLOORDIVI32;
                let outs = self
                    .session
                    .benchmark_run::<i32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_i32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::DIVF32 => {
                // TODO merge logics together and pass the OpCode
                let operand_out = self.get_next_byte() as usize;
                let operand_lhs = self.get_next_byte() as usize;
                let operand_rhs = self.get_next_byte() as usize;
                let lhs_dataview = self.data_buffer_f32.remove(&operand_lhs).unwrap();
                let rhs_dataview = self.data_buffer_f32.remove(&operand_rhs).unwrap();
                // println!("{:?}", lhs_dataview);
                // println!("{:?}", rhs_dataview);
                let opcode = OpCode::DIVF32;
                let outs = self
                    .session
                    .benchmark_run::<f32>(opcode, lhs_dataview, rhs_dataview);
                self.data_buffer_f32.insert(operand_out, outs);
                Ok(0)
            }
            OpCode::CONSTI32 => {
                // TODO do some action, add data_buffer
                // create lhs dataview
                // TODO enable it
                let operand_out = self.get_next_byte() as usize;
                let operand_in = self.get_next_four_bytes();
                let operand_in_i32 = i32::from_le_bytes(operand_in);
                self.push_data_buffer_i32(operand_out, vec![operand_in_i32]);
                Ok(0)
            }
            OpCode::CONSTF32 => {
                // TODO do some action, add data_buffer
                // create lhs dataview
                let operand_out = self.get_next_byte() as usize;
                let operand_in = self.get_next_four_bytes();
                let operand_in_f32 = f32::from_le_bytes(operand_in);
                self.push_data_buffer_f32(operand_out, vec![operand_in_f32]);
                Ok(0)
            }
            OpCode::CONSTTENSOR => {
                Ok(0)
            }
            _ => {
                panic!("Not Implemented Error Execution Step Code");
            }
        }
    }

    // property functions that is public
    pub fn command_buffer(&self) -> &Vec<u8> {
        &self.command_buffer
    }

    // property functions that is public
    pub fn registers(&self) -> &[i32] {
        &self.registers
    }

    pub fn push_bytecode_into_cmdbuffer(&mut self, byte: u8) {
        self.command_buffer.push(byte);
    }

    pub fn get_idata(&self, index: usize) -> &Vec<i32> {
        &self.data_buffer_i32[&index].raw_data
    }

    // TODO to be moved into parametric arguments => push_data<T>(data: Vec<T>)
    pub fn push_data_buffer_i32(&mut self, index: usize, data: Vec<i32>) {
        let mut data_buffer = DataView::<concrete_backend::Backend, i32>::new(
            &self.session.device_context.device,
            &self
                .session
                .device_instance_ref
                .memory_property()
                .memory_types,
            data,
            ElementType::I32,
        );
        self.data_buffer_i32.insert(index, data_buffer);
    }

    pub fn get_fdata(&self, index: usize) -> &Vec<f32> {
        &self.data_buffer_f32[&index].raw_data
    }

    pub fn push_data_buffer_f32(&mut self, index: usize, data: Vec<f32>) {
        let mut data_buffer = DataView::<concrete_backend::Backend, f32>::new(
            &self.session.device_context.device,
            &self
                .session
                .device_instance_ref
                .memory_property()
                .memory_types,
            data,
            ElementType::F32,
        );
        self.data_buffer_f32.insert(index, data_buffer);
    }

    // entry functions for execute, that is public
    pub fn run_once(&mut self) -> Result<u8, RuntimeStatusError> {
        self.step()
    }

    // TODO modify the return into statuscode
    pub fn run(&mut self) -> Result<u8, RuntimeStatusError> {
        println!("start to execute");
        loop {
            if self.program_counter >= self.command_buffer.len() {
                println!("end of execution");
                return Ok(0);
            }
            let status = self.step();
            match status {
                // do nothing if status ok
                Ok(_) => {}
                Err(_) => return status,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_vm_struct() {
        let ist = DeviceInstance::new();
        let vm = VM::new(&ist);
        assert_eq!(vm.registers[0], 0);
    }

    // TODO maybe need to test the middle status when halt invoked in run-until-end way.
    #[test]
    fn test_halt_step() {
        let ist = DeviceInstance::new();
        let mut vm = VM::new(&ist);
        vm.command_buffer = vec![0, 0, 0];
        let exit_code = vm.run_once();
        assert_eq!(exit_code.is_ok(), true);
        let u8_exit_code = exit_code.unwrap();
        assert_eq!(u8_exit_code, 1);
        assert_eq!(vm.program_counter, 1);
    }

    #[test]
    fn test_vm_dummy() {
        let ist = DeviceInstance::new();
        let mut vm = VM::new(&ist);
        vm.command_buffer = vec![];
        let exit_code = vm.run();
        assert_eq!(exit_code.is_ok(), true);
        assert_eq!(vm.program_counter, 0);
    }

    #[test]
    fn test_vm_illegal() {
        let ist = DeviceInstance::new();
        let mut vm = VM::new(&ist);
        vm.command_buffer = vec![255];
        let exit_code = vm.run();
        assert_eq!(exit_code.is_ok(), false);
        assert_eq!(vm.program_counter, 1);
    }

    #[test]
    fn test_vm_fetch_instruction() {
        let ist = DeviceInstance::new();
        let mut vm = VM::new(&ist);
        vm.command_buffer = vec![0];
        let opcode = vm.fetch_instruction();
        assert_eq!(opcode.unwrap(), OpCode::HALT);
    }

    #[test]
    fn test_vm_next_byte() {
        let ist = DeviceInstance::new();
        let mut vm = VM::new(&ist);
        vm.command_buffer = vec![8];
        let data = vm.get_next_byte();
        assert_eq!(data, 8);
    }

    #[test]
    fn test_vm_next_two_bytes() {
        let ist = DeviceInstance::new();
        let mut vm = VM::new(&ist);
        vm.command_buffer = vec![2, 7];
        let data = vm.get_next_two_bytes();
        assert_eq!(data, 519);
    }

    #[test]
    fn test_load_op() {
        // TODO replace with new mock code, not old loadstore ADD system
        assert_eq!(0, 0);
    }

    #[test]
    fn test_mul_op() {
        assert_eq!(0, 0);
    }

    #[test]
    fn test_floordiv_op() {
        assert_eq!(0, 0);
    }

    #[test]
    fn test_vm_push_data_f32() {
        assert_eq!(0, 0);
    }

    // TODO to support int for DataView
    #[test]
    fn test_vm_push_data_i32() {
        assert_eq!(0, 0);
    }
}
