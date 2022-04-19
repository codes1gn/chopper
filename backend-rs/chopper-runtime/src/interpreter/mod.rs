extern crate float_eq;

use float_eq::{assert_float_eq, float_eq};

use nom::types::CompleteStr;
use std;
use std::io;
use std::io::Write;
use std::num::ParseIntError;

use crate::assembler::parse_bytecode;
use crate::base::errors::*;
use crate::instance::*;
use crate::session::*;
use crate::vm::VM;

#[derive(Debug)]
pub struct Interpreter<'a> {
    history: Vec<String>,
    pub vm: VM<'a>,
}

impl<'a> Interpreter<'a> {
    pub fn new(ist: &'a DeviceInstance) -> Interpreter<'a> {
        Interpreter {
            history: vec![],
            vm: VM::new(&ist),
        }
    }

    /*
    pub fn bootstrap(&'a mut self) {
        self.vm = VM::new(&self.hw_instance);
    }

    pub fn vm(&self) -> &mut VM<'a> {
        self.vm.as_ref_mut().unwrap()
    }
    */

    /// Accepts a hexadecimal string WITHOUT a leading `0x` and returns a Vec of u8
    /// Example for a LOAD command: 00 01 03 E8
    /// TODO add this attr, to ensure its deprecated
    // #[allow(dead_code)]
    fn parse_hex(&mut self, i: &str) -> Result<Vec<u8>, ParseIntError> {
        let split = i.split(' ').collect::<Vec<&str>>();
        let mut results: Vec<u8> = vec![];
        for hex_string in split {
            let byte = u8::from_str_radix(hex_string, 16);
            match byte {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        // TODO currently only allow for 4 bytes length format of instruction
        // to be extended later
        assert_eq!(results.len(), 4);
        Ok(results)
    }

    // interface wrapped for tests on interpreter
    pub fn mock_operation(&mut self, cmd_buffer: &str) -> Result<u8, RuntimeStatusError> {
        let status = self.consume_command(cmd_buffer);
        status
    }

    pub fn run_bytecode(&mut self, cmd_buffer: String) -> Result<u8, RuntimeStatusError> {
        let status = self.consume_command(cmd_buffer.as_str());
        status
    }

    fn consume_command(&mut self, cmd_buffer: &str) -> Result<u8, RuntimeStatusError> {
        match cmd_buffer {
            "exit" | "quit" | "q" => {
                println!("Chopper-Runtime Halt Now");
                // TODO make put this setting to base const, halt exit code use 1, else use 0
                Ok(7)
            }
            "history" | "h" => {
                for cmd in &self.history {
                    println!("|-- {}", cmd);
                }
                // TODO history cmd use 6 as status code
                Ok(6)
            }
            "list" | "l" => {
                println!("action: Showing instruction queue");
                for inst in self.vm.command_buffer() {
                    println!("|-- {}", inst);
                }
                // TODO
                Ok(5)
            }
            "display" | "watch" | "wt" => {
                println!("action: Showing registers");
                let mut reg_table = vec![];
                for reg in self.vm.registers() {
                    reg_table.push(reg.clone());
                }
                println!("{:#?}", reg_table);
                // TODO
                Ok(4)
            }
            _ => {
                // parse bytecodes
                let parsed_program = parse_bytecode(CompleteStr(cmd_buffer));
                let (_, result_program) = parsed_program.expect("failed to parse bytecode");
                let bytecode = result_program.to_bytes();
                for byte in bytecode {
                    self.vm.push_bytecode_into_cmdbuffer(byte);
                }
                self.vm.run_once().expect("execution status not correct");
                // TODO return the parse and execute status not handle return here
                // TODO use code 0 temparilly
                Ok(0)
            }
        }
    }

    pub fn run(&mut self) {
        println!("~~~~~~~~~  Entering Chopper Runtime ~~~~~~~~~~");
        loop {
            let mut cmd_buffer = String::new();
            let stdin = io::stdin();

            // show >> prompts
            print!(">> ");
            io::stdout().flush().expect("error: Failed to print");

            // blocking until inputs come
            stdin
                .read_line(&mut cmd_buffer)
                .expect("error: Failed to read user inputs");
            let cmd_buffer = cmd_buffer.trim();
            // after handling this command, add it to the history list
            let step_status = self.consume_command(&cmd_buffer);
            match step_status {
                Ok(exit_code) => (),
                Err(error_code) => {
                    break;
                }
            }
            self.history.push(cmd_buffer.to_string());
        }
        println!("~~~~~~~~ Exiting Chopper Runtime ~~~~~~~~");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_interpreter() {
        //let ipt = Interpreter::new();
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        assert_eq!(ipt.history.len(), 0);
    }

    #[test]
    fn test_push_history() {
        //let mut ipt = Interpreter::new();
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        let fake_cmd = String::from("exit");
        ipt.history.push(fake_cmd.to_string());
        assert_eq!(ipt.history[0], "exit");
    }

    #[test]
    fn test_mock_halt() {
        //let mut ipt = Interpreter::new();
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        let status = ipt.mock_operation("quit");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 7);
    }

    #[test]
    fn test_mock_history() {
        //let mut ipt = Interpreter::new();
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        let status = ipt.mock_operation("history");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 6);
    }

    #[test]
    fn test_mock_list() {
        //let mut ipt = Interpreter::new();
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        let status = ipt.mock_operation("list");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 5);
    }

    #[test]
    fn test_mock_bytecode_halt() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        let status = ipt.mock_operation("halt");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_i32_literal() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        // TODO make runtime check on matching const.i32 and i32 type annotation
        let status = ipt.mock_operation("%17 = crt.literal.const.i32! 13 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(17), vec![13]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_f32_literal() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%8 = crt.literal.const.f32! 1.3 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_fdata(8), vec![1.3]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_add_i32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_idata(1), vec![1]);
        assert_eq!(*ipt.vm.get_idata(2), vec![2]);

        // add
        let status = ipt.mock_operation("%3 = crt.add.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(3), vec![3]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_sub_i32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_idata(1), vec![1]);
        assert_eq!(*ipt.vm.get_idata(2), vec![2]);

        // add
        let status = ipt.mock_operation("%3 = crt.sub.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(3), vec![-1]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_mul_i32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_idata(1), vec![1]);
        assert_eq!(*ipt.vm.get_idata(2), vec![2]);

        // add
        let status = ipt.mock_operation("%3 = crt.mul.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(3), vec![2]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_floordiv_i32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.i32! 1 : i32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_idata(1), vec![1]);
        assert_eq!(*ipt.vm.get_idata(2), vec![2]);

        // add
        let status = ipt.mock_operation("%3 = crt.floordiv.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(3), vec![0]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_floordiv_i32_case2() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.i32! 7 : i32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.i32! 2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_idata(1), vec![7]);
        assert_eq!(*ipt.vm.get_idata(2), vec![2]);

        // add
        let status = ipt.mock_operation("%3 = crt.floordiv.i32! %1, %2 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(3), vec![3]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_add_f32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_fdata(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_fdata(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.mock_operation("%3 = crt.add.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_fdata(3), vec![3.3], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_sub_f32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_fdata(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_fdata(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.mock_operation("%3 = crt.sub.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_fdata(3), vec![-1.1], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_mul_f32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_fdata(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_fdata(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.mock_operation("%3 = crt.mul.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_fdata(3), vec![2.42], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_binary_div_f32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%1 = crt.literal.const.f32! 1.1 : f32\n");
        let status = ipt.mock_operation("%2 = crt.literal.const.f32! 2.2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(*ipt.vm.get_fdata(1), vec![1.1], rmax_all <= 0.00001);
        assert_float_eq!(*ipt.vm.get_fdata(2), vec![2.2], rmax_all <= 0.00001);

        // add
        let status = ipt.mock_operation("%3 = crt.div.f32! %1, %2 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_fdata(3), vec![0.5], rmax_all <= 0.00001);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_f32_binary_add_then_sub_i32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%8 = crt.literal.const.i32! 3 : i32\n");
        let status = ipt.mock_operation("%7 = crt.literal.const.i32! 2 : i32\n");
        let status = ipt.mock_operation("%1 = crt.literal.const.i32! 7 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_idata(8), vec![3]);
        assert_eq!(*ipt.vm.get_idata(7), vec![2]);

        // add
        let status = ipt.mock_operation("%4 = crt.add.i32! %8, %7 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_eq!(*ipt.vm.get_idata(4), vec![5]);

        // sub
        let status = ipt.mock_operation("%5 = crt.sub.i32! %1, %4 : i32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        // TODO package this assert macro into utils, hide rmax_all setting from hardcode
        assert_eq!(*ipt.vm.get_idata(5), vec![2]);
    }

    #[test]
    // TODO fix integer end2end pipeline
    fn test_mock_bytecode_f32_binary_add_then_sub_f32() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%8 = crt.literal.const.f32! 1.3 : f32\n");
        let status = ipt.mock_operation("%7 = crt.literal.const.f32! 2.9 : f32\n");
        let status = ipt.mock_operation("%1 = crt.literal.const.f32! 7.4 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_eq!(*ipt.vm.get_fdata(8), vec![1.3]);
        assert_eq!(*ipt.vm.get_fdata(7), vec![2.9]);

        // add
        let status = ipt.mock_operation("%4 = crt.add.f32! %8, %7 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(*ipt.vm.get_fdata(4), vec![4.2], rmax_all <= 0.00001);

        // sub
        let status = ipt.mock_operation("%5 = crt.sub.f32! %1, %4 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        // TODO package this assert macro into utils, hide rmax_all setting from hardcode
        assert_float_eq!(*ipt.vm.get_fdata(5), vec![3.2], rmax_all <= 0.00001);
    }

    #[test]
    fn test_mock_bytecode_tensor_add() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation(
            "%0 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>\n",
        );
        let status = ipt.mock_operation(
            "%1 = crt.literal.const.tensor! dense<[2.2 3.3 3.3 1.1 3.3 2.2], shape=[2 3]>\n",
        );
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(
            *ipt.vm.get_fdata(0),
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_fshape(0), vec![2, 3]);
        assert_float_eq!(
            *ipt.vm.get_fdata(1),
            vec![2.2, 3.3, 3.3, 1.1, 3.3, 2.2],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_fshape(1), vec![2, 3]);

        // add
        let status = ipt.mock_operation("%4 = crt.add.f32! %0, %1 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(
            *ipt.vm.get_fdata(4),
            vec![3.3, 5.5, 6.6, 5.5, 8.8, 8.8],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_mock_bytecode_tensor_sub() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation(
            "%9 = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>\n",
        );
        let status = ipt.mock_operation(
            "%7 = crt.literal.const.tensor! dense<[2.2 3.3 3.3 1.1 3.3 2.2], shape=[2 3]>\n",
        );
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(
            *ipt.vm.get_fdata(9),
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_fshape(9), vec![2, 3]);
        assert_float_eq!(
            *ipt.vm.get_fdata(7),
            vec![2.2, 3.3, 3.3, 1.1, 3.3, 2.2],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_fshape(7), vec![2, 3]);

        // sub
        let status = ipt.mock_operation("%5 = crt.sub.f32! %7, %9 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(
            *ipt.vm.get_fdata(5),
            vec![1.1, 1.1, 0.0, -3.3, -2.2, -4.4],
            rmax_all <= 0.00001
        );
    }

    #[test]
    fn test_mock_bytecode_tensor_matmul() {
        let ist = DeviceInstance::new();
        let mut ipt = Interpreter::new(&ist);
        // ok
        // matmul(3x2, 2x3) => (3x3)
        let status = ipt.mock_operation(
            "%9 = crt.literal.const.tensor! dense<[1. 2. 3. 4. 5. 6.], shape=[2 3]>\n",
        );
        let status = ipt.mock_operation(
            "%7 = crt.literal.const.tensor! dense<[1. 1. 1. 1. 1. 1.], shape=[3 2]>\n",
        );
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // inspect data valid
        assert_float_eq!(
            *ipt.vm.get_fdata(9),
            vec![1., 2., 3., 4., 5., 6.],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_fshape(9), vec![2, 3]);
        assert_float_eq!(
            *ipt.vm.get_fdata(7),
            vec![1., 1., 1., 1., 1., 1.],
            rmax_all <= 0.00001
        );
        assert_eq!(*ipt.vm.get_fshape(7), vec![3, 2]);

        // matmul, temparilly faked with add
        let status = ipt.mock_operation("%5 = crt.matmul.f32! %7, %9 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        assert_float_eq!(
            *ipt.vm.get_fdata(5),
            vec![5., 7., 9., 5., 7., 9., 5., 7., 9.],
            rmax_all <= 0.00001
        );

        let status = ipt.mock_operation(
            "%9 = crt.literal.const.tensor! dense<[1. 2. 3. 4. 5. 6.], shape=[2 3]>\n",
        );
        let status = ipt.mock_operation(
            "%7 = crt.literal.const.tensor! dense<[1. 1. 1. 1. 1. 1.], shape=[3 2]>\n",
        );
        let status = ipt.mock_operation("%6 = crt.matmul.f32! %9, %7 : f32\n");
        assert_float_eq!(
            *ipt.vm.get_fdata(6),
            vec![6., 6., 15., 15.],
            rmax_all <= 0.00001
        );
    }
}
