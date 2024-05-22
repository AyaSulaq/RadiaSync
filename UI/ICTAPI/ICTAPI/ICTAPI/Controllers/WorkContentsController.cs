using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using ICTAPI.ictDB;
using System.Diagnostics;
using Microsoft.AspNetCore.Authorization;
using Microsoft.Extensions.FileProviders;
using System.Net;
using Microsoft.IdentityModel.Tokens;
using Microsoft.CodeAnalysis.Elfie.Diagnostics;
using System.Text;
using System.IO;
using NuGet.Protocol.Plugins;
using System.IO.Compression;
using static ICTAPI.Controllers.WorkContentsController;

namespace ICTAPI.Controllers
{

    [Route("api/[controller]")]
    [ApiController]
    public class WorkContentsController : ControllerBase
    {
        private readonly IctAppContext _context;

        public WorkContentsController(IctAppContext context)
        {
            _context = context;
        }
        public class ApiResponse<T>
        {
            public bool Success { get; set; }
            public string Message { get; set; }
            public T Data { get; set; }
        }


        [HttpGet("user/{userId}")]
        public async Task<IActionResult> GetWorkContentsByUserId(int userId)
        {
            var userWorkContents = await _context.WorkContents.Where(w => w.UserId == userId).ToListAsync();

            if (userWorkContents == null || userWorkContents.Count == 0)
            {
                return NotFound(new ApiResponse<List<WorkContent>> { Success = false, Message = "No work contents found for the user", Data = null });
            }

            return Ok(new ApiResponse<List<WorkContent>> { Success = true, Message = "Work contents retrieved successfully", Data = userWorkContents });
        }





        // GET: api/WorkContents
        [HttpGet]
        public async Task<ActionResult<ApiResponse<IEnumerable<WorkContent>>>> GetWorkContents()
        {
            var workContents = await _context.WorkContents.ToListAsync();
            return Ok(new ApiResponse<IEnumerable<WorkContent>> { Success = true, Message = "Work contents retrieved successfully", Data = workContents });
        }


        [HttpGet("{id}")]
        public async Task<IActionResult> GetWorkContent(int id)
        {
            var workContent = await _context.WorkContents.FindAsync(id);

            if (workContent == null)
            {
                return NotFound(new ApiResponse<object> { Success = false, Message = "Work content not found", Data = null });
            }

            try
            {
                var copiedImageUrls = CopyImagesToWorkFolder(workContent);
                return Ok(new ApiResponse<List<string>> { Success = true, Message = "Work content retrieved successfully", Data = copiedImageUrls });
            }
            catch (Exception ex)
            {
                return Problem(ex.Message);
            }
        }

        private List<string> CopyImagesToWorkFolder(WorkContent workContent)
        {
            var copiedImageUrls = new List<string>();
            var sourceFolderPath = "source/folder/path"; 
            var workFolderPath = "C:\\inetpub\\wwwroot\\publish\\images"; 

            var filePaths = GetAllFilesInPath(workContent);

            foreach (var filePath in filePaths)
            {
                // Generate a unique file name or use existing file name
                var fileName = Path.GetFileName(filePath);
                var destinationPath = Path.Combine(workFolderPath, fileName);

                // Copy the file to the work folder
                System.IO.File.Copy(filePath, destinationPath, true);

                // Generate URL for the copied image
                var imageUrl = $"http://localhost:8080/images/{Path.GetFileName(destinationPath)}"; // Update with your domain
                copiedImageUrls.Add(imageUrl);
            }

            return copiedImageUrls;
        }

        // PUT: api/WorkContents/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutWorkContent(int id, WorkContent workContent)
        {
            if (id != workContent.Id)
            {
                return BadRequest();
            }

            _context.Entry(workContent).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!WorkContentExists(id))
                {
                    return NotFound();
                }
                else
                {
                    throw;
                }
            }

            return NoContent();
        }


        // POST: api/WorkContents
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost]
        [DisableRequestSizeLimit]
        public async Task<ActionResult<ApiResponse<List<(string, string)>>>> PostWorkContent([FromForm] IFormFile formFile, [FromForm] string type, [FromForm]int userId)
        {
                WorkContent workContent = new WorkContent();
            try
            {
                if (_context.WorkContents == null)
                {
                    return Problem("Entity set 'IctAppContext.WorkContents'  is null.");
                }
                if (type == null || string.IsNullOrEmpty(type))
                {
                    return Problem("Type is missing.");
                }

                workContent.CreatedAt = DateTime.Now;

                string basePath = "C:\\Users\\USER\\Desktop";
                string inputPath = Path.Combine(basePath, "input");

                // Check if the input directory exists, create it if not
                if (!Directory.Exists(inputPath))
                {
                    Console.WriteLine("Creating Path");
                    Directory.CreateDirectory(inputPath);
                }
                Console.WriteLine("Path Su");

                if (formFile.Length > 0)
                {
                    try
                    {
                        string filePath = Path.Combine(inputPath, formFile.FileName);

                        using (var fileStream = new FileStream(filePath, FileMode.Create))
                        {
                            await formFile.CopyToAsync(fileStream);
                        }

                        Console.WriteLine("Saving File in " + filePath);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        return Problem(ex.Message);
                    }
                }
                try
                {
                    if (type == "mri")
                    {
                        workContent.OutputPath = mriToct(formFile);
                        Console.WriteLine(workContent.OutputPath);
                    }
                    else if (type == "ct")
                    {
                        workContent.OutputPath = ctTomri(formFile);
                        Console.WriteLine(workContent.OutputPath);
                    }
                }
                catch (Exception ex)
                {
                    return Problem($"{ex.Message}");
                }

                workContent.InputPath = Path.Combine(inputPath, formFile.FileName);
                workContent.FileName = formFile.FileName;
                workContent.UserId = userId;
               // workContent.PatientId = 12;
                workContent.FinishedAt = DateTime.Now;
               _context.WorkContents.Add(workContent);
                await _context.SaveChangesAsync();
              //  Console.WriteLine(await _context.WorkContents.FindAsync());
            }
            catch (Exception e) 
            {
                return Problem(e.Message);
            }
            return Ok(new ApiResponse<List<(string, string)>> { Success = true, Message = "Finished Success", Data = new List<(string, string)> { (workContent.InputPath, workContent.FileName) } });
        }



        private string ctTomri(IFormFile formFile)
        {
            //string pythonCmd = $"python3 ct_to_mri.py /mnt/c/Users/USER/Desktop/input/{formFile.FileName}";
            //string command = $"cd /home/user/FedMed-GAN-main : {pythonCmd} --output_dir /mnt/c/Users/USER/Desktop/output/cttomri";
            //ExecuteCommandInWSL(command, "");
            //return "/mnt/c/Users/User/Desktop/output/cttomri";
            string fileName = Path.GetFileNameWithoutExtension(formFile.FileName);
            string outputPath = $"/mnt/c/Users/USER/Desktop/output/mritoct/{fileName}/";
            ExecuteCommandInWSL(formFile.FileName, outputPath);


            outputPath = $"C:\\Users\\USER\\Desktop\\output\\cttomri\\{fileName}\\";

            return Path.Combine(outputPath);
        }

        private string mriToct(IFormFile formFile)
        {
            string pythonCmd = $"python3 mri_to_ct.py /mnt/c/Users/USER/Desktop/input/{formFile.FileName}";
            string command = $"cd /home/user/FedMed-GAN-main : {pythonCmd} --output_dir /mnt/c/Users/USER/Desktop/output/mritoct";


            string fileName = Path.GetFileNameWithoutExtension(formFile.FileName);
            string outputPath = $"/mnt/c/Users/USER/Desktop/output/cttomri/{fileName}/";
            ExecuteCommandInWSL(formFile.FileName, outputPath);
            //return $"C:\\Users\\USER\\Desktop\\output\\{formFile.Name}";
            outputPath = $"C:\\Users\\USER\\Desktop\\output\\mritoct\\{fileName}\\";

            return Path.Combine(outputPath);
        }

        private void ExecuteCommandInWSL(string fileName, string outputahmad)
        {

            string command;
            Process process = new Process();
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = "cmd.exe";
            command = $"/C wsl python3 /home/user/FedMed-GAN-main/ct_to_mri.py /mnt/c/Users/USER/Desktop/input/{fileName} --output_dir {outputahmad}";
            // command = "ls /mnt/host/c/Users/User";
            startInfo.Arguments = command;//"wsl -e " + command; // $"C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\ahmad.exe.lnk {command}";
            startInfo.UseShellExecute = false;
            startInfo.RedirectStandardOutput = true;
            startInfo.CreateNoWindow = false;
            process.StartInfo = startInfo;
            process.Start();
            Console.WriteLine("Proccess Started");
            Console.WriteLine("Command : ");
            Console.WriteLine(command);
            StreamReader sr = process.StandardOutput;
            string output = sr.ReadToEnd();
            Console.WriteLine("Window Output : ");
            Console.WriteLine(output);
            process.WaitForExit();
            Console.WriteLine("Proccess Exited");

        }

        // DELETE: api/WorkContents/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteWorkContent(int id)
        {
            if (_context.WorkContents == null)
            {
                return NotFound();
            }
            var workContent = await _context.WorkContents.FindAsync(id);
            if (workContent == null)
            {
                return NotFound();
            }

            _context.WorkContents.Remove(workContent);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool WorkContentExists(int id)
        {
            return (_context.WorkContents?.Any(e => e.Id == id)).GetValueOrDefault();
        }

        private List<string> GetAllFilesInPath(WorkContent workList)
        {
            List<string> result = new List<string>();
            try
            {
                var path = workList.OutputPath;
                if (path != null && Directory.Exists(path))
                {
                    string[] files = Directory.GetFiles(path);
                    if (files.Length > 0)
                    {
                        foreach (string file in files)
                        {
                            result.Add(Path.Combine(path, Path.GetFileName(file)));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }
            return result;
        }






    }
}
