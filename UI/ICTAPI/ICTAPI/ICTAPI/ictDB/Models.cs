using Microsoft.Identity.Client;
using System.ComponentModel;

namespace ICTAPI.ictDB;



public partial class LoginModel
{
    public string Email { get; set; } = null!;

    public string Password { get; set; } = null!;

}
